# MiniMax-H3 on g7e（RTX PRO 6000 Blackwell Server Edition）

在 AWS `g7e` 实例（NVIDIA RTX PRO 6000 Blackwell Server Edition，96 GB）上部署和优化
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（33B 联合视频+音频 DiT）的部署脚本。
三种模式全覆盖：t2va / fl2va / ref2va，步数与分辨率都是入参。

**交付方案：单卡 NVFP4 4-bit 权重 + SageAttention**（sglang），对 BF16 快 **1.84–1.97×**，
比上一版交付的 FP8+sage 再快 **1.26–1.29×**（32/32 格全赢）。

| 场景 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| **sglang NVFP4+sage（交付）** | | | | |
| fl2va，1 卡 | 31.125 s | 45.145 s | 114.419 s | 170.278 s |
| ref2va（参考短边 1024），1 卡 | 36.443 s | 53.225 s | 119.050 s | 177.138 s |
| fl2va，2 卡 Ulysses=2 | 25.164 s | 36.686 s | 77.231 s | 115.379 s |
| ref2va（参考短边 1024），2 卡 Ulysses=2 | 28.876 s | 42.324 s | 79.451 s | 118.706 s |
| 加卡加速（ref2va） | 1.26× | 1.26× | **1.50×** | **1.49×** |
| **sglang FP8+sage（上一版交付）** | | | | |
| fl2va，1 卡 | 39.612 s | 58.159 s | 144.392 s | 215.993 s |
| ref2va（参考短边 1024），1 卡 | 45.862 s | 67.744 s | 149.654 s | 223.827 s |
| fl2va，2 卡 Ulysses=2 | 29.302 s | 43.069 s | 87.298 s | 130.732 s |
| ref2va（参考短边 1024），2 卡 Ulysses=2 | 33.622 s | 49.609 s | 89.574 s | 134.169 s |
| 加卡加速（ref2va） | 1.36× | 1.37× | 1.67× | 1.67× |
| **NVFP4+sage 对 FP8+sage** | | | | |
| 1 卡（fl2va / ref2va） | 1.27 / 1.26× | 1.29 / 1.27× | 1.26 / 1.26× | 1.27 / 1.26× |
| 2 卡（fl2va / ref2va） | 1.16 / 1.16× | 1.17 / 1.17× | 1.13 / 1.13× | 1.13 / 1.13× |
| **NVFP4 不叠 sage（消融）** | | | | |
| ref2va，1 卡 | 46.753 s | 68.768 s | 174.065 s | 261.130 s |
| ref2va，2 卡 Ulysses=2 | 33.818 s | 49.836 s | 106.529 s | 160.041 s |
| **NVFP4 4-bit（ComfyUI + sage，2 卡靠 Raylight Ulysses=2）** | | | | |
| ref2va，1 卡 | 35.76 s | 50.97 s | 123.84 s | 178.13 s |
| ref2va，2 卡 Ulysses=2 | 33.42 s | 47.33 s | 99.20 s | 138.93 s |
| NVFP4 加卡加速 | 1.07× | 1.08× | 1.25× | 1.28× |

请求口径：`short_edge` 480/768 + `aspect 16:9`、`duration 5.0`（5.175 s 成片、124 帧、24 fps）、
`flow_shift 12.0`、`audio_flow_shift 3.0`、固定 seed。**加卡只在 768p 划得来**，见「加卡」一节。

### 关于 4-bit（NVFP4）

上表四块都是**同一台 `g7e.12xlarge`、同 prompt/seed/几何**下量的，时间口径也同一个（整条请求：
文本编码 + 采样 + VAE 解码 + 封装）。同批次重测的 FP8+sage 对照格（ref2va 768p/20 单卡）
150.061 s vs 表里 149.654 s，差 0.27%，所以跨块比较不是跨重启漂移造出来的。四条结论：

1. **sglang 的 NVFP4 现在是对的，而且 NVFP4+sage 是这张卡上最快的一档。** 早先写的
   "NVFP4 画质不可用"、"坏的是 w4a4 这个格式"**都已撤回**，真正的两个缺陷是：scale 布局按后端
   字符串分支，sm_120 上被强制 `backend="auto"` 时两个分支都不进（→ 每层相对误差 0.55 而不是
   0.095）；以及 `qkv_proj` 的**行**被 DiT 自己的 loader 置换过而 per-row 块标度没跟着换
   （→ 52 层全拿错标度，一整片灰糊）。两个都在 `scripts/patches/` 里。
2. **纯 NVFP4 在目标档位是退步，必须配 sage**：不叠 sage 时 480p 和 FP8+sage 打平（fl2va 甚至快
   2%），768p 反而慢 15–19%。**权重量化碰不到 attention，而 attention 是 768p 一步的 59%。**
   叠上 sage 后 32/32 格全赢。sage 单独给 NVFP4 的加速：768p 1.45×、480p 1.25×。
3. **2 卡的倍数系统性低于 1 卡**（1.13–1.17 vs 1.26–1.29），加卡收益也从 1.62× 掉到 1.49×。
   Ulysses 的 PCIe 通信是固定开销，DiT 算得越快它占比越大——**任何"让 DiT 更快"的优化都会同时
   稀释加卡收益**，所以报倍数必须说清是几卡。这也是"sage 会让并行效率变差"那条规律的同一个机制。
4. **对客户自己那套 ComfyUI+Raylight，我们 6/8 赢**（1 卡 0.98/0.96/1.04/1.01×，
   2 卡 1.16/1.12/1.25/1.17×），输的两格是 480p 单卡。优势在扩展性：sglang 的 Ulysses 值
   1.48–1.50×，Raylight 只有 1.07–1.28×（原生 ComfyUI 没有任何分布式设施）。

**显存反而涨了**：NVFP4+sage 单卡峰值 52.0 GB，FP8+sage 是 47.7 GB（BF16 79.5 GB）。权重是省了
（文件 24.4 GB vs FP8 的 ~31 GB），但整体峰值高 4.3 GB。**要速度选 NVFP4，要显存余量选 FP8**——
两个都装不下"一张卡上两个副本"（那需要 ≤48 GB），三种模式全上线仍然是一卡一个副本。

## 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/` | **部署要用的 11 个**，见下表 |
| `scripts/patches/` | 5 个 sglang 补丁（480p、width/height、CPU offload、参考短边 env、缺参策略），`serve.sh` 自动应用；外加 2 个 NVFP4 必需的源码补丁（TMA 标度布局、qkv 块标度行序），由 `g7e_nvfp4_table.sh` 应用 |
| `scripts/assets/` | 示例输入素材（都是我们自己生成的 t2va 片子切出来的，可随意用） |
| `scripts/bench/` | 测量工具，只用来复核数字，部署不需要 |
| `scripts/capacity/` | 抢 g7e 机器用的（spot 配额/容量探测），跑在 Jump box 上 |
| `scripts/rejected/` | 被否掉方案的脚本（NVFP4、FA4），留作证据，别照着跑 |

部署只需要顶层这 11 个：

| 脚本 | 用途 |
|---|---|
| `nvfp4_quantize_transformer.py` | 从 stock bf16 权重量化出 NVFP4 checkpoint（~90 s）。**交付路径的第一步** |
| `nvfp4_canonicalize.py` | 把第三方（ComfyUI 转换器）导出的 NVFP4 文件掰成 sglang 期望的布局；只有 Ref2VA 有现成的，FL2VA 得用上面那个自己量化 |
| `build_sageattention.sh` | 在容器里从源码编 SageAttention（sm_120），pip 上的 wheel 只值 1.16× |
| `g7e_nvfp4_table.sh` | 量 NVFP4[+sage] 的 16 格表：两个 variant × 1/2 卡 × 4 个几何，自动打两个 NVFP4 补丁 |
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
./build_sageattention.sh                 # 容器 h3、commit d1a57a5、幂等
docker exec h3 pip show sageattention | head -2      # 应为 2.2.0
```

自检**必须在源码目录之外**跑：在 `/sgl-workspace/SageAttention` 里 `import sageattention` 命中的是
没编好的源码包，报 `cannot import name '_fused' from partially initialized module`——看着像编译失败，
其实 pip 早就 `Successfully installed` 了。脚本已经 `-w /tmp`。

它只装进**这个容器的 site-packages**。容器一删就没了，`--attention-backend sage_attn` 会起不来。
想免掉重编译就固化成镜像：

```bash
docker commit h3 h3-sage:local
# 以后所有 serve.sh 都带上 IMAGE=h3-sage:local
```

**编译期间不要有正在计时的请求**：nvcc 吃满 48 vCPU 会把非 DiT 那几段（VAE 出帧、封装）推上去。
最干净是和下一步的 checkpoint 量化（也是纯 CPU）放在同一个窗口里做。

### 3.5 准备 NVFP4 checkpoint 并打两个补丁（约 3 分钟）

两个 partition 各要一个 checkpoint，来路不同：

```bash
# FL2VA：从 stock bf16 现量化（~90 s，纯 CPU，峰值 host RAM ≈ 文件大小）
docker cp nvfp4_quantize_transformer.py h3:/tmp/
docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer -e DST=/out/nvfp4_fl2va.safetensors \
  h3 python3 /tmp/nvfp4_quantize_transformer.py
# 期望结尾：wrote ...: 951 tensors (q=208 fp8=50 copy=277) worst rel=0.0951

# Ref2VA：lilcheaty/MiniMax-H3-NVFP4 的 *_nvfp4_mixed.safetensors，下完必须规范化
docker cp nvfp4_canonicalize.py h3:/out/
docker exec h3 python3 /out/nvfp4_canonicalize.py     # -> /out/nvfp4_ref2va_fixed.safetensors

# 两个源码补丁（幂等，改的是容器里的 sglang，不走 serve.sh 那套 .patch 流程）
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  docker cp patches/$p h3:/tmp/ && docker exec h3 python3 /tmp/$p
done
```

FL2VA 也可以走第三方，但那边只有裁过的 `_pruned_nvfp4`，和 stock 权重不是一回事，所以自己量化。
两条路出来的配方是同一套：208 个线性层 nvfp4 + 50 个 `adaln_proj.linear.weight` 裸 fp8（占 DiT 权重
40% 但算力 0%，纯显存项）+ 其余 bf16，951 个张量，无 `input_scale`；round-trip 相对误差
0.094–0.0951 = group-16 e2m1 的量化地板，比这个大很多就是打包/标度算错了。

`g7e_nvfp4_table.sh` 会自己做这一整步，所以只想量表的话可以跳过。

### 4. 起服务

一个进程只能服务一个 partition：**fl2va 分区服务 t2va + fl2va（端口 30010）**，
**ref2va 分区只服务 ref2va（端口 30030）**。

```bash
cd scripts
NVFP4ENV="SGLANG_USE_RUNAI_MODEL_STREAMER=0 \
          SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND=auto H3_FP4_TMA_SCALES=1 H3_FP4_QKV_FIX=0"
SAGE="--attention-backend sage_attn --component-attention-backends text_encoder=torch_sdpa"

# t2va + fl2va
VARIANT=fl2va GPUS=1 ULYSSES=1 ENVX="$NVFP4ENV" \
  EXTRA="--layerwise-offload-components text_encoder \
         --transformer-weights-path /out/nvfp4_fl2va.safetensors $SAGE" ./serve.sh start

# ref2va（参考短边 1024 是 1.46× 的杠杆；上游默认是 2048）
VARIANT=ref2va GPUS=1 ULYSSES=1 \
  ENVX="$NVFP4ENV SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" \
  EXTRA="--layerwise-offload-components text_encoder \
         --transformer-weights-path /out/nvfp4_ref2va_fixed.safetensors $SAGE" ./serve.sh start
```

三个 NVFP4 env 和两个源码补丁（第 3.5 步）缺一个就是废片或起不来，见「关于 4-bit」一节。
**别传 `--quantization modelopt_fp4`**：从 safetensors 自动推断的配置才是对的，显式给会建出一个空的
不能用的 config。想退回 FP8 就把 `--transformer-weights-path` 和三个 NVFP4 env 换成
`--quantization fp8`，其余不动。

**三种模式全上线要 2 卡**，一卡一个副本（NVFP4+sage 52.0 GB / FP8+sage 47.7 GB，两个副本挤一张卡
必 OOM）：

```bash
DEVICES=0 VARIANT=fl2va  GPUS=1 ULYSSES=1 ENVX="$NVFP4ENV" \
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path /out/nvfp4_fl2va.safetensors $SAGE" ./serve.sh start
DEVICES=1 VARIANT=ref2va GPUS=1 ULYSSES=1 ENVX="$NVFP4ENV SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" \
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path /out/nvfp4_ref2va_fixed.safetensors $SAGE" ./serve.sh start
```

单条延迟优先、且能接受占满 2 卡跑一个请求时，改用 Ulysses：`GPUS=2 ULYSSES=2`
（NVFP4+sage 768p 实测 **1.49–1.50×**，FP8+sage 是 1.65–1.70×，PCIe P2P 27.5 GB/s 够用；
480p 只有 1.26×，见「加卡」一节）。

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
ATTN=sage_attn ./g7e_nvfp4_table.sh    # 交付配置（NVFP4+sage）的 16 格，1 卡分母在同一次跑里
./g7e_2card_sage.sh                    # FP8+sage：同机 1 卡分母 + 2 卡四个场景 + 画质对比
TASK=ref2va ./g7e_2card_sage.sh        # ref2va（参考短边 1024），要 Ref2VA 分区的服务
CASES="768_30" SKIP_G1=1 ./g7e_2card_sage.sh
```

**交付配置（NVFP4+sage）的加卡收益比下表低一档**：480p 1.23–1.26×、768p 1.48–1.50×
（对应 FP8+sage 的 1.35–1.37× / 1.65–1.67×）。机制就是下面第 2 条的另一面——通信是固定开销，
NVFP4 把分子压小了，分母没动。**所以"加卡值多少"这个数必须绑定量化配置来报。**

同机同镜像实测（FP8+sage，5.175 s 成片）：

| 场景 | 1 卡 | 2 卡 Ulysses=2 | 加速 | 并行效率 | 每步通信 |
|---|---|---|---|---|---|
| fl2va 480p/20 | 39.612 | 29.302 | 1.352× | 67.6% | 0.449 s/步 |
| fl2va 480p/30 | 58.159 | 43.069 | 1.350× | 67.5% | — |
| fl2va 768p/20 | 144.392 | 87.298 | 1.654× | 82.7% | 0.763 s/步 |
| fl2va 768p/30 | 215.993 | 130.732 | 1.652× | 82.6% | — |
| ref2va 480p/20 | 45.862 | 33.622 | 1.364× | 68.2% | 0.505 s/步 |
| ref2va 480p/30 | 67.744 | 49.609 | 1.366× | 68.3% | — |
| ref2va 768p/20 | 149.654 | 89.574 | 1.671× | 83.5% | 0.751 s/步 |
| ref2va 768p/30 | 223.827 | 134.169 | 1.668× | 83.4% | — |

两个 task 的曲线几乎重合，因为 latent 几何相同、搬的字节就相同：**每步通信开销和 task 无关**。

三件事按这个顺序才站得住：

1. **分母必须同机量**。fl2va 这台的 1 卡 768p/30 是 219.762 s、上一台 215.993 s（+1.75%），
   ref2va 那台 227.544 s vs 旧表 223.827 s（+1.66%）——跨重启漂移实测能到 5.4%，拿别的机器的
   单卡数当分母，加速比就是编的。按同机分母是 **1.681×（fl2va）/ 1.696×（ref2va）**。
   注意只有**时间**会漂：同镜像同配置同 seed 的成片跨机器 md5 逐位相同，所以画质参考片可以复用。
2. **480p 加卡是浪费**。每步通信 0.449 s 占了 2 卡每步 1.377 s 的 33%，所以只有 1.35×。
   768p 序列长（seq≈39760），同样的字节摊薄到更大的算力上，才有 1.65–1.67×。
3. **画质变化要跟自己的对照组比，不能跟 1.0 比**。fl2va：SSIM 0.971、运动能量 +1.3%、
   码率 +0.3% → 只是加法顺序变了。ref2va 的 2 卡成片运动能量掉 **21%**（0.4675→0.3678），
   但把 sage 关掉、只留 Ulysses 的对照组掉得一样多（−21.2%），所以这是 Ulysses 换了扩散轨迹、
   不是量化损伤。**ref2va 要成片和单卡逐帧一致就只能 1 卡**。

通信开销按 20/30 步两点斜率反推（2 卡实际斜率 − 单卡斜率的一半）。量化会改这个数：
BF16 是 **1.091 s/步**，FP8 把 KV 字节减半后是 **0.722 s/步**，**sage 不动它**（0.763，差在噪声内）
——sage 只压 attention 算力、不改搬运字节。所以 FP8+sage 的效率比纯 FP8 低：分子被压小了，
分母那块固定开销没变。ref2va 同机对照组把这条第二次量了出来：768p/30 纯 FP8 是
312.634 → 175.320（**1.784× / 89.2%**），加上 sage 是 227.544 → 134.169（**1.696× / 84.8%**）。
**sage 在 2 卡上仍值 1.30×**（fl2va 168.643 → 130.732，ref2va 175.320 → 134.169）。

**成本**：`g7e.12xlarge` 的每卡·小时比单卡机型便宜约 8%（同刻东京 1a spot $3.8676/h ÷ 2 =
$1.9338 vs `g7e.4xlarge` $2.1035），但不到 100% 的并行效率把这点便宜吃掉还倒亏 —— 768p/30
每成片秒 fl2va 贵 **11.3%**（$0.0244 → $0.0271）、ref2va 贵 **9.0%**（$0.0257 → $0.0280）。
**加卡买的是延迟，不是单价。** spot 折扣是浮动的，每次报价都要重查当刻两个机型的价格。

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
9. **NVFP4 的 qkv 行序修复只能有一处生效。** 库里给的是
   `patches/patch_h3_qkv_scale_reorder.py`；如果容器里还装了别的、在 `process_weights_after_loading`
   里做同一件事的补丁，**两个一起开就是 reorder 两次，和一次都不做一样坏**（同样是一片灰糊）。
   `H3_FP4_QKV_FIX=0` 就是关掉另一处的开关，加着不会有副作用。
   同理 `H3_FP4_TMA_SCALES=1` 只有在 `patch_nvfp4_tma_scale_layout.py` 打上之后才有意义——
   **两个 env 都是静默的**：忘了打补丁不会报错，只会出废片，所以第一次跑完必须看成片。
10. **g7e 按小时计费，用完就 terminate。** G 系 spot 配额默认只有 64 vCPU，抢不到多卡是配额问题
   不是容量问题；容量在 us-east-1 / eu-central-1 更好。

## 被否掉的方案

- ~~**在 sglang 上跑 NVFP4**~~：**已翻案，现在是交付方案**，见「关于 4-bit」。当时看到的废片
  （SSIM 0.585/0.671、码率 4.2×、运动能量 1.764）是两个 sglang 缺陷叠加，不是 w4a4 这个格式的问题；
  两个补丁在 `scripts/patches/`。留这条是记着教训：**"格式不行"这个结论当时下得太早了**，
  两次撤回（先撤"坏的是 4-bit 激活"，再撤"坏的是 w4a4 本身"）都是因为消融把待测的那条路径绕开了
  （反量化成 bf16 = 不走那个 fp4 GEMM；只比权重误差 = 不过 loader 的行置换）。
- **DiT cache / 跳步**（TeaCache 等三套）：层次上接不上 H3。
- **FlashAttention-4**：sm_120 两条路全封。
- **裸算力优化**：单卡已在 roofline 上——attention 实测 368.7 TFLOPS（峰值 409.9 的 90%）、四个线性层
  403–414 TFLOPS（100%），trace 里非 matmul 只占 4.1%。所以能动的只有 attention 的**数值精度**
  （= SageAttention），这也是为什么它是唯一有效的杠杆。
