# MiniMax-H3 on g7e（RTX PRO 6000 Blackwell Server Edition）

在 AWS `g7e` 实例（NVIDIA RTX PRO 6000 Blackwell Server Edition，96 GB）上部署和优化
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（33B 联合视频+音频 DiT）的部署脚本。
三种模式全覆盖：t2va / fl2va / ref2va，步数与分辨率都是入参。

**最佳方案：单卡 FP8 权重 + SageAttention**，对 BF16 快 **1.46–1.54×**，峰值显存 79.5 → 48.8 GB，
画质明显好于"换一个 seed"的下限。**没有把权重量化到 4-bit**：公开的 NVFP4 checkpoint 是 w4a4，画质不可用。

| 场景 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| fl2va，1 卡 | 39.612 s | 58.159 s | 144.392 s | 215.993 s |
| ref2va（参考短边 1024），1 卡 | 45.862 s | 67.744 s | 149.654 s | 223.827 s |
| fl2va，2 卡 Ulysses=2 | 29.302 s | 43.069 s | 87.298 s | 130.732 s |

请求口径：`short_edge` 480/768 + `aspect 16:9`、`duration 5.0`（5.175 s 成片、124 帧、24 fps）、
`flow_shift 12.0`、`audio_flow_shift 3.0`、固定 seed。**加卡只在 768p 划得来**，见「加卡」一节。

## 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/` | **部署要用的 7 个**，见下表 |
| `scripts/patches/` | 5 个 sglang 补丁（480p、width/height、CPU offload、参考短边 env、缺参策略），`serve.sh` 自动应用 |
| `scripts/assets/` | 示例输入素材（都是我们自己生成的 t2va 片子切出来的，可随意用） |
| `scripts/bench/` | 测量工具，只用来复核数字，部署不需要 |
| `scripts/capacity/` | 抢 g7e 机器用的（spot 配额/容量探测），跑在 Jump box 上 |
| `scripts/rejected/` | 被否掉方案的脚本（NVFP4、FA4），留作证据，别照着跑 |

部署只需要顶层这 7 个：

| 脚本 | 用途 |
|---|---|
| `g7e_bringup.sh` | 裸机 → 可起服务（下权重 + 拉镜像），必须 detached 跑 |
| `serve.sh` | 起/停/查服务，自动建容器并打补丁（幂等）。**核心** |
| `h3gen.py` | 发请求的客户端；三种模式、步数分辨率全是入参 |
| `g7e_arm.sh` | 跑单个优化 arm（固定口径，配对基线写在脚本头部）；加旋钮时用它 |
| `quality_pair.sh` | 画质校验（在机器上跑）：SSIM + 运动能量 + 码率 |
| `quality_pair_local.sh` | 同上，但在笔记本上对已下载的 mp4 跑 |
| `g7e_2card_sage.sh` | 量「加卡值多少」：同机 1 卡分母 + 2 卡 Ulysses=2，跑完自动对画质 |

`scripts/bench/` 里的 7 个是复核用的：`attn_bench_sage.py`（attention 单点，H3 真实形状
`seq=41456`）、`attn_bench.py`、`gemm_bench.py`（这张卡的实测峰值）、`prof_step.sh` +
`trace_summary.py`（一步按 kernel 拆开，61.6% 那个数就是它出的）、`gap_probe.sh`、
`patch_sage_kernel_select.py`（挑 sage kernel，只做消融）。

成片、服务端日志、逐次运行结果和完整工作记录都不在这个库里。这里只放跑起来需要的东西。

## 部署最佳方案

前提：`g7e.4xlarge`（1×96 GB）或 `g7e.12xlarge`（2 卡），DLAMI，NVMe 挂在 `/opt/dlami/nvme`。
容器镜像 `lmsysorg/sglang:dev`，**实测通过的是 sglang `273d978bed`**（torch 2.13.0+cu130，nvcc 13.0）。

⚠️ **`:dev` 是移动标签。** 2026-08-14 重跑一次 bringup 就把它从 `273d978bed` 挪到了 `c4271c3fe1`，
验证过的镜像变成 dangling（一次 `docker image prune` 就没了）。正在跑的容器不受影响（它有自己的
文件系统），但**新建**的容器会落在新 commit 上，那里 4 个 git 补丁未验证、sageattention 还得重编。
所以验证过的镜像要打个不会被回收的标签，并在 `serve.sh` 里钉住：

```bash
docker tag <验证过的镜像ID> lmsysorg/sglang:h3-validated
IMAGE=lmsysorg/sglang:h3-validated ./serve.sh start     # 以后都这么起
```

### 1. 裸机准备（约 1.5 小时，主要在下权重）

```bash
setsid nohup ./scripts/g7e_bringup.sh > ~/bringup.log 2>&1 < /dev/null &
```

装 `python3.12-venv` + `ffmpeg`、下权重到 `/opt/dlami/nvme/h3`、并行拉镜像。
**两个分区都下**（`Ref2VA/` + `FL2VA/`，共 269 GB），因为三种模式分属两个分区、一个进程只能加载一个。
`Ref2VA/` 先下，所以它下完就能先起 ref2va 服务，不用等 FL2VA。只要一半就
`PARTS="Ref2VA" ./scripts/g7e_bringup.sh`（135 GB）。

权重目录必须**叫** `MiniMax-H3`（`serve.sh` 挂成 `/models/MiniMax-H3`）：sglang 从
`--model-path` 的 basename 反解 pipeline 类，改个名就 `module diffusers has no attribute
MiniMaxH3ModularPipeline`。

### 2. 建容器并打补丁

第 3 步编译 SageAttention 需要容器已经存在，所以先单独把容器和补丁做出来（**0.07 秒**）：

```bash
cd scripts
./serve.sh prepare
```

`serve.sh` 建的是一个 `sleep infinity` 的常驻容器 `h3`，并幂等地打上 5 个补丁；`prepare` 只跑
这半段，不起服务、不加载权重。之后换 flag 重启服务不会重拉镜像、不会重打补丁。
（`DRYRUN=1 ./serve.sh` 可以先只看摆放。）

`stop` 故意不会创建容器——它只杀服务进程、保留容器（打好的补丁和容器里 pip 装的东西都在里面，
所以停服务不会丢 sage）。

### 3. 在容器里编译 SageAttention（约 12 分钟）

**必须在容器内编译**——服务跑在容器里，而 pip 上的 `sageattention==1.0.6` 是纯 Triton 版本，
只有 1.16×，拿不到我们测的 1.33×。

```bash
docker exec h3 bash -lc '[ -d /tmp/SageAttention ] || git clone https://github.com/thu-ml/SageAttention /tmp/SageAttention; \
  cd /tmp/SageAttention && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=24 EXT_PARALLEL=4 \
  NVCC_APPEND_FLAGS="--threads 8" pip install --no-build-isolation .'
docker exec h3 pip show sageattention | head -2      # 应为 2.2.0
```

写成 `[ -d ] ||` 是因为 `cd /tmp && git clone … && pip install …` 这种写法在**已经编过一次**的机器上
会因为目录已存在而 clone 失败，`&&` 把整条链短路掉，看着像出错其实什么也没干。

它只装进**这个容器的 site-packages**。容器一删就没了，`--attention-backend sage_attn` 会起不来。
想免掉重编译就固化成镜像：

```bash
docker commit h3 h3-sage:local
# 以后所有 serve.sh 都带上 IMAGE=h3-sage:local
```

### 4. 起服务

一个进程只能服务一个 partition：**fl2va 分区服务 t2va + fl2va（端口 30010）**，
**ref2va 分区只服务 ref2va（端口 30030）**。

```bash
cd scripts
FP8SAGE="--layerwise-offload-components text_encoder --quantization fp8 \
         --attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa"

# t2va + fl2va
VARIANT=fl2va GPUS=1 ULYSSES=1 \
  ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0" \
  EXTRA="$FP8SAGE" ./serve.sh start

# ref2va（参考短边 1024 是 1.46× 的杠杆；上游默认是 2048）
VARIANT=ref2va GPUS=1 ULYSSES=1 \
  ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" \
  EXTRA="$FP8SAGE" ./serve.sh start
```

**三种模式全上线要 2 卡**，一卡一个副本（48.8 GB 正好半张卡；两个副本挤一张卡是
97710/97887 MiB，必 OOM）：

```bash
DEVICES=0 VARIANT=fl2va  GPUS=1 ULYSSES=1 ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0" EXTRA="$FP8SAGE" ./serve.sh start
DEVICES=1 VARIANT=ref2va GPUS=1 ULYSSES=1 ENVX="SGLANG_USE_RUNAI_MODEL_STREAMER=0 SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" EXTRA="$FP8SAGE" ./serve.sh start
```

单条延迟优先、且能接受占满 2 卡跑一个请求时，改用 Ulysses：`GPUS=2 ULYSSES=2`
（768p 实测 **1.65–1.68×**，PCIe P2P 27.5 GB/s 够用；480p 只有 1.35×，见「加卡」一节）。

`./serve.sh status | logs | stop` 默认作用于**所有**副本；只想操作一个就带上 `VARIANT=`。

### 5. 发请求

```bash
python3 h3gen.py --task ref2va --image assets/first.png --inline \
  --short-edge 768 --aspect 16:9 --duration 5.0 --steps 20 \
  --seed 1234 --flow-shift 12.0 --audio-flow-shift 3.0 \
  --prompt "..." --port 30030 --out myclip
```

`--short-edge` + `--aspect` 和 `--width/--height` 二选一（后者需要 width/height 补丁，
不能和 `--wire` 混用）。步数、分辨率都是入参。成片落在宿主机
`/opt/dlami/nvme/out/videos`。

扫一遍上面那张表的 4 个几何（同 seed 才能配对比较）：

```bash
for edge in 480 768; do for st in 20 30; do
  python3 h3gen.py --task ref2va --image assets/first.png --inline \
    --short-edge $edge --aspect 16:9 --duration 5.0 --steps $st \
    --seed 1234 --flow-shift 12.0 --audio-flow-shift 3.0 \
    --prompt "..." --port 30030 --out ref2va_${edge}_${st}
done; done
```

要试一个新旋钮别手搓 serve + 请求，用 `g7e_arm.sh <tag> [额外 flag...]`：它按固定口径
（ref2va / 1344×768 / 参考短边 1024 / 20 和 30 步）重起服务、发请求、打印 `inference_time_s`
和峰值显存，并回读日志确认 attention backend 真的换了（**这一步不能省**，见坑 1）：

```bash
./g7e_arm.sh fp8_sage --quantization fp8 --attention-backend sage_attn \
  --component-attention-backends text_encoder=torch_sdpa
```

同口径的 BF16 配对基线写在该脚本头部（20 步 224.05 s / 30 步 337.61 s / 峰值 88108 MiB），
所以一个 arm 跑完就能直接算倍数。

### 6. 校验画质

```bash
RUNDIR=$HOME/h3run/scripts ./quality_pair.sh <BF16参考名> <候选名>
```

**必须三个指标一起看**：SSIM 只回答"采样有没有变"，跳步类优化会把运动能量压塌却让 SSIM 上升，
量化类会加噪却可能不动运动能量。判据是三条实测下限：单卡同 seed 重跑 **SSIM 1.000000**（逐位相同），
**2 卡 Ulysses 的归约序地板 0.944 量级**（all-to-all 改加法顺序，同 seed 也回不到 1.0），
只换 seed **0.56–0.60**。落在地板以上且运动能量/码率不出圈的就是轨迹微扰，不是画质坏。

## 加卡（2 卡 Ulysses=2）

```bash
./g7e_2card_sage.sh                    # 同机 1 卡分母 + 2 卡四个场景 + 画质对比
CASES="768_30" SKIP_G1=1 ./g7e_2card_sage.sh
```

同机同镜像实测（FP8+sage，fl2va，5.175 s 成片）：

| 场景 | 1 卡 | 2 卡 Ulysses=2 | 加速 | 并行效率 | 每步通信 |
|---|---|---|---|---|---|
| 480p/20 | 39.612 | 29.302 | 1.352× | 67.6% | 0.449 s/步 |
| 480p/30 | 58.159 | 43.069 | 1.350× | 67.5% | — |
| 768p/20 | 144.392 | 87.298 | 1.654× | 82.7% | 0.763 s/步 |
| 768p/30 | 215.993 | 130.732 | 1.652× | 82.6% | — |

三件事按这个顺序才站得住：

1. **分母必须同机量**。这台的 1 卡 768p/30 是 219.762 s，上一台是 215.993 s（+1.75%）——跨重启
   漂移实测能到 5.4%，拿别的机器的单卡数当分母，加速比就是编的。按同机分母是 **1.681× / 84.0%**。
2. **480p 加卡是浪费**。每步通信 0.449 s 占了 2 卡每步 1.377 s 的 33%，所以只有 1.35×。
   768p 序列长（seq≈39760），同样的字节摊薄到更大的算力上，才有 1.65×。
3. **画质是归约序差异，不是损伤**：SSIM 0.971、运动能量 +1.3%、码率 984k→987k（+0.3%）。
   在 0.944 的地板之上、码率不动 → 只是加法顺序变了。

通信开销按 20/30 步两点斜率反推（2 卡实际斜率 − 单卡斜率的一半）。量化会改这个数：
BF16 是 **1.091 s/步**，FP8 把 KV 字节减半后是 **0.722 s/步**，**sage 不动它**（0.763，差在噪声内）
——sage 只压 attention 算力、不改搬运字节。所以 FP8+sage 的效率（84.0%）比纯 FP8（87.3%）低：
分子被压小了，分母那块固定开销没变。**sage 在 2 卡上仍值 1.29×**（168.643 → 130.732）。

**成本**：`g7e.12xlarge` 的每卡·小时比单卡机型便宜约 8%（同刻东京 1a spot $3.8676/h ÷ 2 =
$1.9338 vs `g7e.4xlarge` $2.1035），但 82.6% 的效率把这点便宜吃掉还倒亏 ——
0.919 × 2 ÷ 1.652 = **1.113**，即 768p/30 每成片秒贵 11.3%（$0.0244 → $0.0271）。
**加卡买的是延迟，不是单价。**

## 已知的坑

1. **`--component-attention-backends transformer=sage_attn` 对 H3 的 DiT 完全无效，而且静默**
   ——测出来就是纯 BF16 的数。per-component 后端是只在组件加载期存活的 contextvar，而 H3 的 DiT
   首次 forward 才解析 backend。必须用全局 `--attention-backend sage_attn`。
2. **全局 sage 必须给 text encoder 豁免** `text_encoder=torch_sdpa`：Qwen3VL 的 `LocalAttention`
   只认 `{fa, torch_sdpa}`，否则起服务就死。
3. **`SGLANG_USE_RUNAI_MODEL_STREAMER=0` 不能省**：Run:ai streamer 会攒匿名内存把主机打爆，
   关掉走 mmap。
4. **`--attention-backend fa` 在 sm_120 会被静默降级**（gate 在 `_FlashAttentionBackendResolver`），
   Sage 的 resolver 没有这个 gate，所以 sage 是真的生效了。
5. 单卡 96 GB 装不下 63 GiB text encoder + 62 GiB DiT，`--layerwise-offload-components
   text_encoder` 是必须项而非选项。它是加载期摆放，encoder 只在 denoise 前跑一次（0.65 s）。
6. `serve.sh` 的默认 warmup 是 `"1344x768 864x480"`；只跑 480p 可以 `WARMUP="864x480"` 省启动时间。
7. 别在 ssh 的命令行里写 `pkill -f <脚本名>` 这种模式——**它会匹配到自己**，
   我们踩过两次（一次 ssh exit 255，一次等待循环空转 10 分钟）。用 PID。
8. **别在已经跑通的机器上重跑 `g7e_bringup.sh`**：下载是幂等的（秒回 0 字节），但 `docker pull`
   会把 `:dev` 挪到新 sglang，见上面那条 ⚠️。要补下另一半权重就直接用 `hf download`。
9. **g7e 按小时计费，用完就 terminate。** G 系 spot 配额默认只有 64 vCPU，抢不到多卡是配额问题
   不是容量问题；容量在 us-east-1 / eu-central-1 更好。

## 被否掉的方案

- **NVFP4 w4a4**：修好 3 处布局偏差后速度 1.26× 是真的，但 SSIM 0.585/0.671、码率 4.2×、
  运动能量 1.764 —— 废片。消融证明**坏的是 4-bit 激活不是权重**（同权重 w4a16 = 0.9375），
  且权重误差 0.0942 已等于理想 group-16 RTN 的 0.0938，**文件侧无可修**。sglang 没有 w4a8 中间档。
- **DiT cache / 跳步**（TeaCache 等三套）：层次上接不上 H3。
- **FlashAttention-4**：sm_120 两条路全封。
- **裸算力优化**：单卡已在 roofline 上——attention 实测 368.7 TFLOPS（峰值 409.9 的 90%）、四个线性层
  403–414 TFLOPS（100%），trace 里非 matmul 只占 4.1%。所以能动的只有 attention 的**数值精度**
  （= SageAttention），这也是为什么它是唯一有效的杠杆。
