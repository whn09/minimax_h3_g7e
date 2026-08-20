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
| fl2va，1 卡 | 38.821 s | 56.769 s | 165.992 s | 248.980 s |
| fl2va，2 卡 Ulysses=2 | 28.996 s | 42.541 s | 102.512 s | 153.927 s |
| ref2va，1 卡 | 46.753 s | 68.768 s | 174.065 s | 261.130 s |
| ref2va，2 卡 Ulysses=2 | 33.818 s | 49.836 s | 106.529 s | 160.041 s |
| **NVFP4 4-bit（ComfyUI + sage，2 卡靠 Raylight Ulysses=2）** | | | | |
| ref2va，1 卡 | 35.76 s | 50.97 s | 123.84 s | 178.13 s |
| ref2va，2 卡 Ulysses=2 | 33.42 s | 47.33 s | 99.20 s | 138.93 s |
| NVFP4 加卡加速 | 1.07× | 1.08× | 1.25× | 1.28× |
| **交付方案再叠 Cache-DiT（可选加速档，有损）** | | | | |
| fl2va，1 卡 | 19.039 s | 20.719 s | 70.312 s | 77.004 s |
| ref2va（参考短边 1024），1 卡 | 20.418 s | 24.044 s | 73.409 s | 86.008 s |
| fl2va，2 卡 Ulysses=2 | 14.968 s | 16.353 s | 47.444 s | 52.023 s |
| ref2va（参考短边 1024），2 卡 Ulysses=2 | 15.716 s | 18.655 s | 48.733 s | 57.264 s |
| 对同口径 base 的倍数（4 个配置的范围） | 1.64–1.83× | 2.18–2.27× | 1.62–1.63× | 2.06–2.22× |
| **再换 Turbo LoRA 蒸馏权重，8 步（最省档，有损）** | 480p/8 | | 768p/8 | |
| fl2va，1 卡 | 14.581 s | | 47.315 s | |
| fl2va，1 卡 + Cache-DiT `RDT=0.24` | **11.882 s** | | **36.400 s** | |
| fl2va，2 卡 Ulysses=2 | 11.242 s | | 31.502 s | |
| fl2va，2 卡 + Cache-DiT `RDT=0.24` | **8.983 s** | | **24.020 s** | |
| ref2va（参考短边 1024），1 卡 + Cache-DiT `RDT=0.24` | **13.513 s** | | **37.727 s** | |
| ref2va（参考短边 1024），2 卡 + Cache-DiT `RDT=0.24` | **10.077 s** | | **24.651 s** | |
| 对同分辨率 stock 20 步的倍数（1 卡 / 1 卡+cache / 2 卡+cache） | 2.14× / **2.62×** / 3.47× | | 2.41× / **3.14×** / **4.75×** | |

请求口径：`short_edge` 480/768 + `aspect 16:9`、`duration 5.0`（5.175 s 成片、124 帧、24 fps）、
`flow_shift 12.0`、`audio_flow_shift 3.0`、固定 seed。**加卡只在 768p 划得来**，见「加卡」一节。

前四块是**无损**的（同卡数同 seed 逐位可复现），可以互相直接比。最后一块 Cache-DiT 是**有损**的，
所以不进默认交付配置，单独列出来给「愿意用画质换成本」的场景；它需要 sglang ≥ `c0b6474`
（旧版本 H3 的 residual 读出来是 0），档位是 480p `RDT=0.20` / 768p `RDT=0.16`，
画质数字、等成本对照（对减步数）和 $/成片秒都在「Cache-DiT」一节。
最后一块换的是**权重本身**（Turbo LoRA 蒸馏，8 步），损得比 Cache-DiT 多一点但省得也多，
见「Turbo LoRA」一节 —— 那一节还有 **10 s / 15 s 长片**的曲线（15 s 768p 单卡 190.6 s、
2 卡 109.8 s，都不 OOM）和"耗时对片长为什么是超线性但远不到平方"的实测拆解。
ref2va 两行的 NVFP4 权重现在是**我们自己转的**（`g7e_quant.sh`），与早先用的第三方文件运行时
逐 MiB、逐 0.004 s 相同，见 §3.5。

### 关于 4-bit（NVFP4）

上表前四块都是**同一台 `g7e.12xlarge`、同 prompt/seed/几何**下量的，时间口径也同一个（整条请求：
文本编码 + 采样 + VAE 解码 + 封装）。同批次重测的 FP8+sage 对照格（ref2va 768p/20 单卡）
150.061 s vs 表里 149.654 s，差 0.27%，所以跨块比较不是跨重启漂移造出来的。四条结论：

1. **sglang 的 NVFP4 现在是对的，而且 NVFP4+sage 是这张卡上最快的一档。** 早先写的
   "NVFP4 画质不可用"、"坏的是 w4a4 这个格式"**都已撤回**，真正的两个缺陷是：scale 布局按后端
   字符串分支，sm_120 上被强制 `backend="auto"` 时两个分支都不进（→ 每层相对误差 0.55 而不是
   0.095）；以及 `qkv_proj` 的**行**被 DiT 自己的 loader 置换过而 per-row 块标度没跟着换
   （→ 52 层全拿错标度，一整片灰糊）。两个都在 `scripts/patches/` 里。
2. **纯 NVFP4 在目标档位是退步，必须配 sage**：不叠 sage 时 480p 和 FP8+sage 打平（fl2va 甚至快
   2%），768p 反而慢 15–17%（16 格里最差的是 ref2va 768p/30 的 +16.7%）。**权重量化碰不到
   attention，而 attention 是 768p 一步的 59%。** 叠上 sage 后 32/32 格全赢。sage 单独给 NVFP4
   的加速（1 卡，同批次两行相除）：**768p 1.45–1.47×、480p 1.25–1.29×**。
3. **2 卡的倍数系统性低于 1 卡**（1.13–1.17 vs 1.26–1.29），加卡收益也从 1.62–1.63×（NVFP4
   不叠 sage）掉到 1.48–1.50×（叠 sage）。
   Ulysses 的 PCIe 通信是固定开销，DiT 算得越快它占比越大——**任何"让每个 block 更快"的优化都会
   同时稀释加卡收益**，所以报倍数必须说清是几卡。这也是"sage 会让并行效率变差"那条规律的同一个机制。
   **例外是 Cache-DiT**：它跳掉的是整个 block，那一层的 all-to-all 跟着一起跳，所以加卡收益不降
   反微升（8/8 格），见「Cache-DiT」一节。
4. **对照的那套 ComfyUI+Raylight，我们 6/8 赢**（1 卡 0.98/0.96/1.04/1.01×，
   2 卡 1.16/1.12/1.25/1.17×），输的两格是 480p 单卡。优势在扩展性：sglang 的 Ulysses 值
   1.48–1.50×，Raylight 只有 1.07–1.28×（原生 ComfyUI 没有任何分布式设施）。

**显存反而涨了**：NVFP4+sage 单卡峰值 52.0 GB，FP8+sage 是 47.7 GB（BF16 79.5 GB）。权重是省了
（文件 24.4 GB vs FP8 的 ~31 GB），但整体峰值高 4.3 GB。**要速度选 NVFP4，要显存余量选 FP8**——
两个都装不下"一张卡上两个副本"（那需要 ≤48 GB），三种模式全上线仍然是一卡一个副本。

## 目录结构

| 路径 | 内容 |
|---|---|
| `scripts/` | **部署与测量要用的 22 个** + `Dockerfile`，见下表 |
| `scripts/patches/` | 5 个 sglang 补丁（480p、width/height、CPU offload、参考短边 env、缺参策略），`serve.sh` 自动应用；外加 2 个 NVFP4 必需的源码补丁（TMA 标度布局、qkv 块标度行序），由 `g7e_nvfp4_table.sh` 应用；再加 1 个只在测 sol_attn 时要打的（dense 回退换 sage，见「稀疏 attention」一节）；`inplace_ref_short_edge.sh` 是参考短边那一行的 in-place 版（`serve.sh` 和 `Dockerfile` 共用，理由写在文件头） |
| `scripts/assets/` | 示例输入素材（都是我们自己生成的 t2va 片子切出来的，可随意用） |
| `scripts/bench/` | 测量工具，只用来复核数字，部署不需要 |
| `scripts/capacity/` | 抢 g7e 机器用的（spot 配额/容量探测），跑在 Jump box 上 |
| `scripts/rejected/` | 被否掉方案的脚本（早期 NVFP4、FA4、SageAttention 3），留作证据，别照着跑 |

顶层这 22 个（外加 `Dockerfile`）：

| 脚本 | 用途 |
|---|---|
| `Dockerfile` + `build_image.sh` | **把对原版镜像的 8 处运行时改动一次性烤成交付镜像**（见 §1.5）；`Dockerfile` 头部写了为什么值得这么做 |
| `g7e_quant.sh` | **交付路径的第一步**：两个 partition 都从 stock bf16 自己量化出 NVFP4（各约 10 min，纯 CPU） |
| `nvfp4_quantize_transformer.py` | 上面那个脚本调的量化器（配方与自检写在文件头） |
| `nvfp4_canonicalize.py` | 历史参考：把第三方（ComfyUI 转换器）导出的 NVFP4 掰成 sglang 期望的布局。现在两个 partition 都自量化，不需要它 |
| `build_sageattention.sh` | 在容器里从源码编 SageAttention（sm_120），pip 上的 wheel 只值 1.16×（用 §1.5 的镜像就不需要它） |
| `g7e_levers.sh` | 旋钮消融（BCG，1 卡和 2 卡），每个 arm 自带同 session 的 base 分母 + 回读后端 |
| `g7e_nvfp4_table.sh` | 量 NVFP4[+sage] 的 16 格表：两个 variant × 1/2 卡 × 4 个几何，自动打两个 NVFP4 补丁 |
| `g7e_bringup.sh` | 裸机 → 可起服务（下权重 + 拉镜像），必须 detached 跑 |
| `serve.sh` | 起/停/查服务，自动建容器并打补丁（幂等）。**核心** |
| `h3gen.py` | 发请求的客户端；三种模式、步数分辨率全是入参 |
| `g7e_arm.sh` | 跑单个优化 arm（固定口径，配对基线写在脚本头部）；加旋钮时用它 |
| `quality_pair.sh` | 画质校验（在机器上跑）：SSIM + 运动能量 + 码率 |
| `quality_pair_local.sh` | 同上，但在笔记本上对已下载的 mp4 跑 |
| `g7e_2card_sage.sh` | 量「加卡值多少」：同机 1 卡分母 + 2 卡 Ulysses=2，跑完自动对画质 |
| `g7e_ref_edge_sweep.sh` | 扫 ref2va 的参考图短边（1024/768/512），**有损**方向，跑完自动对画质 |
| `g7e_dev_levers.sh` | 在最新 sglang（`:dev`）上做旋钮消融（Cache-DiT / adaln），另起容器 `h3n`，见「最新 sglang」一节 |
| `queue_tabfill.sh` | 用上面那个把 Cache-DiT 推荐档在 4 个配置（fl2va/ref2va × 1/2 卡）上量齐，每组 base 与 cache 同 session |
| `lora_merge_transformer.py` | 把 Turbo LoRA **离线**合进 bf16 transformer（259/259 模块必须全命中），再走同一套 NVFP4 量化，见「Turbo LoRA」一节 |
| `g7e_turbo.sh` | Turbo LoRA 那一轮的四个 phase：stock 参考 / 步数曲线 / 低 RDT（不触发）/ 高 RDT 扫描 |
| `g7e_ref2va_provenance.sh` | 判定 ref2va 的 NVFP4 该用谁转的：自量化 / 第三方+canonicalize / BF16 真值三条腿，见 §3.5 |
| `sol_attn_micro.py` | Sol-Attn 的孤立微基准（H3 DiT 形状，不用起 server）：扫 tau × 片长，复算保留块比例，见「稀疏 attention」一节 |
| `pull_results_loop.sh` | 在笔记本上边跑边拉结果（spot 会被回收，别等跑完再拉） |

`scripts/bench/` 里的 7 个是复核用的：`attn_bench_sage.py`（attention 单点，H3 真实形状
`seq=41456`）、`attn_bench.py`、`gemm_bench.py`（这张卡的实测峰值）、`prof_step.sh` +
`trace_summary.py`（一步按 kernel 拆开，61.6% 那个数就是它出的）、`gap_probe.sh`、
`patch_sage_kernel_select.py`（挑 sage kernel，只做消融）。

成片、服务端日志、逐次运行结果和完整工作记录都不在这个库里。这里只放跑起来需要的东西。

## 部署最佳方案

前提：`g7e.4xlarge`（1×96 GB）或 `g7e.12xlarge`（2 卡），DLAMI，NVMe 挂在 `/opt/dlami/nvme`。
容器镜像从 `lmsysorg/sglang:dev` 拉，但**所有脚本默认用的是固定标签
`lmsysorg/sglang:h3-validated`**（`g7e_bringup.sh` 拉完自动打这个标签，已存在时不覆盖）。
**实测通过的是 sglang `273d978bed`**（torch 2.13.0+cu130，nvcc 13.0）。

⚠️ **`:dev` 是移动标签。** 2026-08-14 重跑一次 bringup 就把它从 `273d978bed` 挪到了 `c4271c3fe1`，
验证过的镜像变成 dangling（一次 `docker image prune` 就没了）。正在跑的容器不受影响（它有自己的
文件系统），但**新建**的容器会落在新 commit 上，那里 4 个 git 补丁未验证、sageattention 还得重编。
这就是所有脚本钉固定标签的原因；手工补建：

```bash
docker tag <验证过的镜像ID> lmsysorg/sglang:h3-validated
IMAGE=lmsysorg/sglang:dev ./serve.sh start     # 只有确实要追 HEAD 时才这么写
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

### 1.5 建交付镜像（约 2 分钟，推荐——它替掉下面第 2 / 3 步和 3.5 里的两个补丁）

到这一步为止对镜像的改动已经有 8 处（2 个 `.patch` + 3 个 python 补丁 + 1 个 in-place 编辑 +
SageAttention + 3 个 env），分散在 `serve.sh`、`build_sageattention.sh` 和几个 driver 里。
`scripts/Dockerfile` 把它们一次性烤进镜像：

```bash
cd scripts
./build_image.sh                 # -> h3-g7e:local，同时打 h3-g7e:<sglang sha7>
IMAGE=h3-g7e:local ./serve.sh start   # 后面每次 serve.sh 都带上 IMAGE
```

三件事值得知道：

- **base 按 digest 钉死**（`lmsysorg/sglang:nightly-dev-20260818-c0b6474b`），不是移动标签，
  所以下面那条 `:dev` 警告在这条路上不适用。要追 HEAD：`BASE=lmsysorg/sglang:dev ./build_image.sh`，
  但**换 base 必须重新确认补丁清单**——这条 RUN 是 `git apply` 失败即中断。
- **`serve.sh` 自己发现清单**：镜像里写了 `/sgl-workspace/.h3-image-patches`（烤了哪些 `.patch`）和
  `/sgl-workspace/.h3-patches/<名字>` 印戳。没有印戳的话，`serve.sh` 的 apply 循环会因为"补丁打不上"
  在一个**已经正确**的镜像上拒绝启动。用镜像起服务时 `PATCHES` 不用传。
- **sm_120 cubin 是建镜像时的断言**：`cuobjdump --list-elf _qattn_sm89*.so | grep sm_120` 必须命中。
  pip 的 wheel 装出来文件名一模一样、只是没有这份 cubin，运行时静默回落 Triton（1.16× 而不是 1.776×）。
  编译不需要 GPU（`TORCH_CUDA_ARCH_LIST=12.0` 交叉编），`MAX_JOBS=8` 下实测 117.6 s。

镜像 47.3 GB。冒烟（全新容器、零手工补丁、768p turbo 8 步 5 s）：**47.242 s**，md5 与打补丁那条路
逐字节相同；480p 8 步 14.621 s（参考 14.581 s，在 0.3% 同进程地板内）。逐位相同就是这条路
"等价于原来那 8 处改动"的判据，不是"看起来差不多"。

不烤进去的：权重（269 GB，照旧 bind mount）、NVFP4 checkpoint（在 `/out`，见 3.5）、GPU 数与并行度
（`serve.sh` 的入参）。

### 2. 建容器并打补丁

（**用了 1.5 的镜像就跳过第 2、3 步和 3.5 里的两个 python 补丁**，只需要跑 `g7e_quant.sh` 出
checkpoint。）第 3 步编译 SageAttention 需要容器已经存在，所以先单独把容器和补丁做出来（**0.07 秒**）：

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

（`docker commit` 出来的东西没人知道里面烤了什么——这就是 §1.5 那个 Dockerfile 存在的原因，
它带 label、带补丁清单、带 sm_120 cubin 断言。）

**编译期间不要有正在计时的请求**：nvcc 吃满 48 vCPU 会把非 DiT 那几段（VAE 出帧、封装）推上去。
最干净是和下一步的 checkpoint 量化（也是纯 CPU）放在同一个窗口里做。

### 3.5 准备 NVFP4 checkpoint 并打两个补丁（约 3 分钟）

**两个 partition 都从 stock bf16 自己量化**，不用任何第三方文件：

```bash
./g7e_quant.sh          # FL2VA + Ref2VA，各约 10 min，纯 CPU
                        # -> /out/nvfp4_fl2va.safetensors、/out/nvfp4_ref2va.safetensors
# 期望结尾：wrote ...: 951 tensors (q=208 fp8=50 copy=277) worst rel=0.09xx

# 两个源码补丁（幂等，改的是容器里的 sglang，不走 serve.sh 那套 .patch 流程）
for p in patch_nvfp4_tma_scale_layout.py patch_h3_qkv_scale_reorder.py; do
  docker cp patches/$p h3:/tmp/ && docker exec h3 python3 /tmp/$p
done
```

配方：208 个线性层 nvfp4 + 50 个 `adaln_proj.linear.weight` 裸 fp8（占 DiT 权重 40% 但算力 0%，
纯显存项）+ 其余 bf16，951 个张量，无 `input_scale`；round-trip 相对误差 0.094±0.002 =
group-16 e2m1 的量化地板，比这个大很多就是打包/标度算错了。两个补丁修的是 sglang 的加载/GEMM
路径，与 checkpoint 来路无关，任何 NVFP4 文件都要打。

**历史**：Ref2VA 早先用的是 lilcheaty 的 `MiniMax_H3_Ref2VA_nvfp4_mixed.safetensors` 过
`nvfp4_canonicalize.py`（→ `nvfp4_ref2va_fixed.safetensors`），因为 FL2VA 那边第三方只有裁过的
`_pruned_nvfp4`、逼着我们写了量化器；量化器与 partition 无关，B300 上 `runquant.sh` 两个都是
自量化的，那半张 ref2va NVFP4 表（16 格，2.28–2.33× vs BF16）全出自自量化文件。所以现在统一
自量化：少一道 canonicalize、无第三方来源问题。`nvfp4_canonicalize.py` 只作历史参考
（要复现旧 `*_fixed` 文件时才用）。

**确认跑已完成**（`g7e_ref2va_provenance.sh`，同机同 session、ref2va、seed 8201、参考短边 1024、
1 卡 sage、20 步）——三条腿里两个 NVFP4 文件在运行时**没有区别**：

| 臂 | 480p | 768p | 峰值显存 |
|---|---|---|---|
| BF16（真值） | 59.675 s | 164.631 s | 78487 MiB |
| 自量化 | **36.277 s**（1.645×） | **118.479 s**（1.390×） | 52383–52399 MiB（−33.3%） |
| 第三方 + canonicalize | 36.281 s | 118.500 s | 同上（逐 MiB 相同） |

差 0.004 s / 0.021 s。画质**只能目视**（3×3 抽帧网格 + 三列 hstack 对照片）：ref2va 的参考视频自带
镜头平移，任何数值扰动都让平移相位错开，逐像素 SSIM 立刻崩 —— bf16→自量化 0.5547/0.7435、
bf16→第三方 0.6741/0.6531、**自量化→第三方 0.6270/0.6869**，三份两两等距，所以这组 SSIM 度量的是
相位差不是画质（同样 NVFP4、fl2va 静止机位那轮是 0.92–0.94）。目视三列同构同细节。
**结论：交付用自量化，第三方文件和 canonicalize 这一步都可以去掉。**

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
         --transformer-weights-path /out/nvfp4_ref2va.safetensors $SAGE" ./serve.sh start
```

三个 NVFP4 env 和两个源码补丁（第 3.5 步）缺一个就是废片或起不来，见「关于 4-bit」一节。
（README 顶部那张 ref2va 表是在旧的 `nvfp4_ref2va_fixed.safetensors` 上量的；自量化文件配方相同、
在 B300 上已用满，g7e 上换过来后建议先跑一格 `quality_pair.sh` 对齐。）
**别传 `--quantization modelopt_fp4`**：从 safetensors 自动推断的配置才是对的，显式给会建出一个空的
不能用的 config。想退回 FP8 就把 `--transformer-weights-path` 和三个 NVFP4 env 换成
`--quantization fp8`，其余不动。

**三种模式全上线要 2 卡**，一卡一个副本（NVFP4+sage 52.0 GB / FP8+sage 47.7 GB，两个副本挤一张卡
必 OOM）：

```bash
DEVICES=0 VARIANT=fl2va  GPUS=1 ULYSSES=1 ENVX="$NVFP4ENV" \
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path /out/nvfp4_fl2va.safetensors $SAGE" ./serve.sh start
DEVICES=1 VARIANT=ref2va GPUS=1 ULYSSES=1 ENVX="$NVFP4ENV SGLANG_MINIMAX_H3_REF_IMAGE_SHORT_EDGE=1024" \
  EXTRA="--layerwise-offload-components text_encoder --transformer-weights-path /out/nvfp4_ref2va.safetensors $SAGE" ./serve.sh start
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

**成本见下一节。** 结论不变且更硬：加卡买的是延迟，不是单价。

## 成本（3 年 Savings Plan 口径）

价格用 `g7e.48xlarge`（8 卡）的 **EC2 3-Year No-Upfront Instance Savings Plans $14.31835/h**，
即 **$1.78979 / 卡·小时**（$0.000497165 / 卡·秒）。这是**统一每卡单价**，1 卡和 2 卡不再有机型
差价，所以下表的 2 卡列直接是"卡数 × 时间"。成片 5.175 s。

交付配置（NVFP4+sage），**每成片秒**成本：

| 场景 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| fl2va 1 卡 | **$0.002990** | **$0.004337** | **$0.010993** | **$0.016360** |
| ref2va 1 卡 | $0.003501 | $0.005113 | $0.011435 | $0.017019 |
| fl2va 2 卡 Ulysses=2 | $0.004836（+61.7%） | $0.007049（+62.5%） | $0.014842（+35.0%） | $0.022171（+35.5%） |
| ref2va 2 卡 Ulysses=2 | $0.005549（+58.5%） | $0.008134（+59.1%） | $0.015268（+33.5%） | $0.022813（+34.0%） |

**每条片子**（fl2va 1 卡）：480p/20 $0.01547、480p/30 $0.02245、768p/20 $0.05689、768p/30 $0.08466。

统一单价下**加卡的溢价就精确等于 1/并行效率 − 1**（768p 效率 74–75% → +33–35%，480p 62–63% →
+59–63%），以前那 8% 的双卡机型折扣不存在了，所以「加卡买延迟不买单价」这条比之前更硬。

1 QPS（每秒一条请求）的机队，按 1 卡 1 副本、fl2va：

| 档位 | 单卡产出 | 需要卡数 | `g7e.48xlarge` 台数 | $/h |
|---|---|---|---|---|
| 768p/20 | 1 条 / 114.4 s | 115 | 14.4 | **$206** |
| 768p/30 | 1 条 / 170.3 s | 171 | 21.4 | $306 |
| 480p/20 | 1 条 / 31.1 s | 32 | 4.0 | **$57** |
| 480p/30 | 1 条 / 45.1 s | 46 | 5.8 | $82 |

**480p 比 768p 便宜 3.7 倍**，档位选择比任何旋钮都值钱。用 DP 副本（一卡一个）而不是 Ulysses
凑吞吐：sglang 对同副本的并发请求是排队串行的，见下节。

对照 B300（`p6-b300.48xlarge` spot $51.7052/h = $6.46315/卡·小时）：768p/20 单卡每成片秒
$0.017040、Ulysses=8 $0.019690。**g7e 的 3 年 SP 每成片秒便宜 1.55×**，但单条延迟 B300 快
2.33×（49.1 vs 114.4 s）、8 卡 Ulysses 能压到 7.1 s。买单价选 g7e，买延迟选 B300。

## 最新 sglang（c0b6474，2026-08-17）实测

镜像身份先说清：交付用的 `lmsysorg/sglang:h3-validated` = `0.0.0.dev1+g273d978be`（2026-08-12），
`lmsysorg/sglang:dev` = `0.0.0.dev1+gc0b6474b4`（2026-08-17），**在 g7e 这台上是两个不同 digest**
（tag 可变，谁后 pull 谁不一样，别拿别的机器上的等号当结论）。跑法见
`scripts/g7e_dev_levers.sh`（另起容器 `h3n`，两个容器不能同时起 server）。

**换镜像本身零收益**（NVFP4+sage，1 卡，fl2va，同 seed）：

| 场景 | 273d978be | c0b6474 | 差 |
|---|---|---|---|
| 768p/20 | 114.484 s | 115.102 s | +0.54% |
| 480p/20 | 31.336 s | 31.44 s | +0.33% |

两个 NVFP4 python 补丁在 c0b6474 上**仍然都要打、也都还能打**（上游没修 sm_120 的标度布局分支）。

**唯一有收益的旋钮是 Cache-DiT**：20 步 1.35–1.92×、30 步 2.0–2.4× 可调，
**有损但在等成本口径下优于减步数**。它不是
`--cache-dit-config`（那条只喂 diffusers 后端），H3 走 native pipeline，旋钮全在 env
（`multimodal_gen/envs.py:46-65`）：`SGLANG_CACHE_DIT_ENABLED` +
`_FN/_BN/_WARMUP/_RDT/_MC/_TAYLORSEER`，`_SECONDARY_*` 给音频那条 transformer。
c0b6474 才修对 H3 的 residual input preservation，旧镜像上读出来是 0。

`_RDT` 是"两步之间 residual 差多小就复用"的阈值，是**唯一值得调的那个**（MC 留默认 3、Fn=1 Bn=0
W=4）。全曲线（1 卡 NVFP4+sage、fl2va、20 步、同 session；SSIM 参考是同分辨率的 base 20 步，
单卡同 seed 重跑逐位相同所以地板是 1.000000）：

| 臂 | 768p 时间 | 倍数 | 768p SSIM Y | motion | 480p 时间 | 倍数 | 480p SSIM Y | motion |
|---|---|---|---|---|---|---|---|---|
| base 20 步 | 113.983 s | 1.00 | — | 0.4372 | 31.236 s | 1.00 | — | 0.3439 |
| RDT 0.12 | 81.305 s | 1.40 | 0.9320 | −2.1% | 23.124 s | 1.35 | 0.9353 | −1.7% |
| **RDT 0.16** | **70.312 s** | **1.62** | **0.9311** | −5.2% | 20.395 s | 1.53 | 0.9327 | −0.3% |
| **RDT 0.20** | 64.896 s | 1.76 | 0.9242 | −11.5% | **19.039 s** | **1.64** | **0.9352** | +0.6% |
| RDT 0.24（默认） | 59.485 s | 1.92 | 0.9215 | −9.6% | 16.341 s | 1.91 | 0.9098 | +6.1% |
| RDT 0.04 | 114.997 s | 1.00 | 1.000000 | — | 31.448 s | 1.00 | 1.000000 | — |
| *减步数对照* base 12 步 | 69.564 s | 1.64 | 0.9099 | −5.1% | 20.137 s | 1.55 | 0.9198 | +1.0% |
| *减步数对照* base 10 步 | 58.281 s | 1.96 | 0.8999 | −3.2% | 17.375 s | 1.80 | 0.9142 | +3.5% |

**30 步上加速比更大**（同素材同口径，参考是同分辨率的 base 30 步）：

| 臂 | 768p/30 | 倍数 | SSIM Y | motion | 480p/30 | 倍数 | SSIM Y | motion |
|---|---|---|---|---|---|---|---|---|
| base 30 步 | 169.601 s | 1.00 | — | 0.4954 | 45.093 s | 1.00 | — | 0.3607 |
| RDT 0.12 | 98.778 s | 1.72 | 0.9280 | −7.0% | 26.143 s | 1.72 | 0.9310 | −3.9% |
| **RDT 0.16** | **77.004 s** | **2.20** | **0.9176** | −20.2% | 22.073 s | 2.04 | 0.9249 | −8.3% |
| **RDT 0.20** | 77.006 s | 2.20 | 0.9176 | −20.2% | **20.719 s** | **2.18** | **0.9219** | −8.9% |
| RDT 0.24（默认） | 71.560 s | 2.37 | 0.9156 | −20.5% | 20.713 s | 2.18 | 0.9219 | −8.9% |

**RDT 0.16 与 0.20 在 fl2va 768p/30 上输出逐格相同**（0.20/0.24 在 fl2va 480p/30 上同样相同）——
gate 命中同一批步，别把它们当两档报。这是**素材/模式相关的巧合，不是通用规律**：同一格在 2 卡上
也重现（52.023 vs 52.012 s、SSIM 六位全同），但换到 ref2va 就不成立（768p/30 两档差 5.6 s）。
步数越多可复用的 residual 越多，所以 30 步的倍数（2.0–2.4×）明显高于 20 步（1.4–1.9×）。

**等成本对照是这一节的重点**（同样的钱，Cache-DiT vs base 减步数）：

| 分辨率 | 步数 | 同成本一对 | Cache-DiT | 减步数 | ΔSSIM |
|---|---|---|---|---|---|
| 768p | 30 | ~98 s | RDT 0.12 → **0.9280** | 17 步 97.4 s → 0.9021 | **+0.026** |
| 768p | 30 | ~76 s | RDT 0.16 → **0.9176 @ 77.0 s** | 13 步 75.0 s → 0.8851 | **+0.032** |
| 768p | 20 | ~70 s | RDT 0.16 → **0.9311** | 12 步 → 0.9099 | +0.021 |
| 768p | 20 | ~59 s | RDT 0.24 → **0.9215** | 10 步 → 0.8999 | +0.022 |
| 480p | 30 | ~26 s | RDT 0.12 → **0.9310** | 16 步 25.7 s → 0.9066 | **+0.024** |
| 480p | 30 | ~21 s | RDT 0.20 → **0.9219 @ 20.7 s** | 13 步 21.5 s → 0.9006 | **+0.021**（还快 0.8 s） |
| 480p | 20 | ~20 s | RDT 0.20 → **0.9352 @ 19.04 s** | 12 步 → 0.9198 @ 20.14 s | +0.015（还快 1.1 s） |
| 480p | 20 | ~17 s | RDT 0.24 → 0.9098 @ 16.34 s | 10 步 → 0.9141 @ 17.38 s | −0.004（打平） |

30 步上 6/6 全赢、3 组同时更快更好；20 步只有最省的 ~1.9× 那档打平。

**对交付最有用的一条：30 步 + Cache-DiT 比现在的 20 步基线又快又好。** 拿 base 30 步（= 想要的画质
目标）当共同参考：

| 分辨率 | base 20 步 | Cache-DiT 30 步 | 结论 |
|---|---|---|---|
| 768p | 113.983 s / 0.9109 | **77.004 s**（RDT 0.16）/ **0.9176** | 快 **1.48×**、SSIM 高 0.007 |
| 480p | 31.236 s / 0.9175 | **20.719 s**（RDT 0.20）/ **0.9219** | 快 **1.51×**、SSIM 高 0.004 |

**推荐档位：768p `_RDT=0.16`、480p `_RDT=0.20`**（20 步 1.62×/1.64×，30 步 2.20×/2.18×）。
**代价要说清**：cache 掉的运动能量比减步数多（768p/30 −20.2% vs 减步数 −11~16%），
cache 复用的正是 residual、掉的正是帧间变化，所以不能只看 SSIM。本轮素材（近静止机位）目视看不出，
大幅运镜/快动作的输入要重新判一次。

**上面那两张曲线表是 fl2va 单卡的；推荐档在四个配置上都量过**（`queue_tabfill.sh`，
每个配置的 base 与 cache 在同一次调用内跑完，倍数只在自己那组里算）：

| 配置 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| fl2va，1 卡 | 19.039 s (1.64×) | 20.719 s (2.18×) | 70.312 s (1.62×) | 77.004 s (2.20×) |
| ref2va，1 卡 | 20.418 s (1.79×) | 24.044 s (2.21×) | 73.409 s (1.62×) | 86.008 s (2.06×) |
| fl2va，2 卡 | 14.968 s (1.68×) | 16.353 s (2.25×) | 47.444 s (1.63×) | 52.023 s (2.22×) |
| ref2va，2 卡 | 15.716 s (1.83×) | 18.655 s (2.27×) | 48.733 s (1.63×) | 57.264 s (2.07×) |

同一轮里三个配置的 **base** 与已发布交付表 12/12 格差 ≤0.2%（而交付表是在旧镜像上量的），所以这两块
可以并列看。**加速比不吃 variant 也不吃卡数**：ref2va 768p/30 的 2.06× 略低于 fl2va 的 2.20×，
是因为它的分母里多了 5–7 s 参考图编码这个 cache 碰不到的固定开销，不是 cache 在 ref2va 上更差。

**Cache-DiT 是唯一不稀释加卡收益的加速手段**（8/8 格不降反微升）：

| 加卡加速 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| fl2va base → +cache | 1.243 → **1.272×** | 1.228 → **1.267×** | 1.474 → **1.482×** | 1.469 → **1.480×** |
| ref2va base → +cache | 1.266 → **1.299×** | 1.256 → **1.289×** | 1.497 → **1.506×** | 1.491 → **1.502×** |

这和本文档反复讲的那条规律（"让 DiT 更快就会稀释加卡收益"，sage 把加卡从 1.62 压到 1.50）
**方向相反**，机制不同：sage 只让每个 block 算得更快，Ulysses 的 all-to-all 一次不少；
Cache-DiT 跳掉的是**整个 block**，那一层的 all-to-all 跟着一起跳，通信和算力同比例下降。
所以 cache 和加卡可以叠，不互相吃。

**ref2va 的 SSIM 不能用，只能目视**（1 卡与 2 卡都一样）：ref2va 的输出继承参考片的运镜，base 自身
motion 就有 1.72–2.04（fl2va 只有 0.36），逐像素 SSIM 对全局位移极敏感 → 量出来 480p 0.64–0.78、
768p 0.88–0.94，**分辨率越低分越低**，与 fl2va 的规律反着走，这是指标失效的签名（NVFP4 provenance
那轮已经踩过同一个坑）。fl2va 2 卡的 SSIM 是可用的，和单卡同量级（768p/30 0.9217 vs 0.9176），
即**加卡不额外损画质**。

- `_RDT=0.04` 是**纯空转**：SSIM 逐格 1.000000 = 一次都没触发，别把它当"保守档"报。
- 加速比和画质**都与内容有关**，这类表只能在真实输入图上量。早先"480p 掉 24% 运动能量"是在
  夜景样例图上配白猫 prompt 跑的（全局重打光把帧间残差顶到极高，gate 行为和正常片不是一回事），
  **已撤回**。同一批的 base 时间反而与内容无关（两张图差 0.4%）。
- 每次都要从 serve 日志回读 `Enabling cache-dit on transformer with config: Fn=1, Bn=0, W=4,
  R=0.24, MC=3 ...`；看到 `Acceleration hooks is disabled for: BlockAdapter` 就是没挂上。
- **请求里带 `quality` 字段会把它整个关掉**（`stages/denoising.py:407`
  `... and "quality" not in explicit_fields`），所以别加 `h3gen.py --quality`。
  而 `quality="high"` 自带的那组审过参数 fail-close 在写死的部署门（4×H200 / 50 步 / sm_9.0），
  g7e 永远进不去。
- 与 `--dit-layerwise-offload` 互斥。

成本口径（$0.000497165/卡·秒 = `g7e.48xlarge` 3年SP $14.31835/h ÷ 8；成片 5.175 s），每成片秒：

| 配置 | 480p/20 | 480p/30 | 768p/20 | 768p/30 |
|---|---|---|---|---|
| fl2va，1 卡 base → cache | $0.003001 → **$0.001829** | $0.004332 → **$0.001990** | $0.010950 → **$0.006755** | $0.016294 → **$0.007398** |
| ref2va，1 卡 base → cache | $0.003508 → **$0.001962** | $0.005106 → **$0.002310** | $0.011418 → **$0.007052** | $0.016998 → **$0.008263** |
| fl2va，2 卡 base → cache | $0.004830 → $0.002876 | $0.007055 → $0.003142 | $0.014854 → $0.009116 | $0.022188 → $0.009996 |
| ref2va，2 卡 base → cache | $0.005540 → $0.003020 | $0.008129 → $0.003584 | $0.015253 → $0.009364 | $0.022796 → $0.011003 |

省 38–46%（20 步）/ 51–56%（30 步）。两条结论：
**30 步 + Cache-DiT 比 20 步 base 更便宜**（fl2va 768p $0.007398 vs $0.010950，便宜 32.4%；
ref2va $0.008263 vs $0.011418，便宜 27.6%），而画质离 30 步的目标更近；
**单卡仍然是最便宜的一档**，加卡买的是延迟不是单价，这一点 cache 开不开都一样。

**`--minimax-h3-adaln-online` 与量化互斥，实测报错**：
`ValueError: MiniMax H3 AdaLN cache is only compatible with unquantized weights`
（`runtime/models/dits/minimax_h3.py:1468`，条件是 `quant_config is not None`）。它省的 24.2 GiB
adaln_proj 是 BF16 口径，NVFP4 下这部分本来就量化过，省不到那么多，凑不出一卡两副本。

**c0b6474 会打死我们的全局 sage 配置**：新镜像里 `audio_vae` 也去 selector 解 attention 后端，
而它只声明 `['fa','torch_sdpa']` → `selector.py:300 ValueError` →
`Failed to load customized audio_vae` → `Rank 0 scheduler is dead`。三个非 DiT 组件一律豁免，
**逗号分隔**：

```bash
--attention-backend sage_attn \
--component-attention-backends text_encoder=torch_sdpa,audio_vae=torch_sdpa,video_vae=torch_sdpa
```

## Turbo LoRA（8 步蒸馏权重，最省档）

[`larryvrh/MiniMax-H3-Turbo-Lora`](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) 的
`minimax_h3_turbo_v4_step600_ema.safetensors`（strength 1.0，模型卡说 4–8 步可用、8 步最好）。
**NVFP4 / sage / Cache-DiT 三个都能叠**，因为我们不走运行时 LoRA 而是**离线合并**。

### 关键做法：离线 bf16 合并，再走原来那套量化

`--lora-path` 那条路在量化权重上是死的：运行时"合并式" LoRA 按 `[out, in]` 做 in-place add
（`runtime/layers/lora/linear.py:243`），而 fp8 把权重转置存（fc1 直接报 `21504 vs 5376`）、
NVFP4 是 `[N, K/2]` 的 packed e2m1 + 块标度，形状永远对不上。`--lora-merge-mode dynamic` 能跑，
但只快 3%、SSIM 0.9346 跌破 2 卡重跑地板 0.9444。

所以先在 bf16 上合并（`scripts/lora_merge_transformer.py`），再拿合并后的目录走 §3.5 同一个
`nvfp4_quantize_transformer.py`。**LoRA 的键名与 diffusers checkpoint 1:1 对得上**，不需要任何映射：
`blocks.N.attn.qkv_proj` / `mlp.fc{1,2}` / `adaln_proj.linear` / `token_refiner.blocks.N.*` /
`final_layer.adaln_proj.linear`，共 259 个模块（518 张量）。`W_eff = W + strength·(B@A)`，
alpha = rank ⇒ 没有额外缩放。合并脚本对"有一个模块没命中"直接 exit 1，不静默少合。

```bash
curl -sL -o /opt/dlami/nvme/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/resolve/main/minimax_h3_turbo_v4_step600_ema.safetensors
docker exec -e SRC=/models/MiniMax-H3/FL2VA/transformer \
  -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  -e DST=/out/turbo_v4_600_bf16 h3n python3 /tmp/lora_merge_transformer.py
#   → merged 259/259 modules, 最大 |delta|/|W| = 0.0036
docker exec -e SRC=/out/turbo_v4_600_bf16 -e DST=/out/nvfp4_fl2va_turbo.safetensors \
  h3n python3 /tmp/nvfp4_quantize_transformer.py
#   → 951 tensors (q=208 fp8=50 copy=277) worst rel=0.0951（stock 是 0.094x，同量级）
```

合并完它就是一份普通 NVFP4 checkpoint，serve 命令与 §4 完全一样，只换 `--transformer-weights-path`。

### 实测（单卡 fl2va，参考 = 同轮 stock NVFP4 20 步）

| 臂 | 480p s | 倍数 | $/成片秒 | SSIM Y | motion | 768p s | 倍数 | $/成片秒 | SSIM Y | motion |
|---|---|---|---|---|---|---|---|---|---|---|
| stock 20 步（参考） | 31.137 | 1.00× | $0.002991 | — | 0.3439 | 114.155 | 1.00× | $0.010967 | — | 0.4372 |
| turbo 4 步 | 9.069 | 3.43× | $0.000871 | 0.8742 | 0.3704 | 25.086 | 4.55× | $0.002410 | 0.8531 | 0.4207 |
| turbo 6 步 | 11.826 | 2.63× | $0.001136 | 0.8867 | 0.3694 | 36.245 | 3.15× | $0.003482 | 0.8695 | 0.4286 |
| turbo 8 步 | 14.581 | 2.14× | $0.001401 | 0.9027 | 0.3558 | 47.315 | 2.41× | $0.004546 | 0.8839 | 0.5193 |
| **8 步 + cache `RDT=0.24` `W=2`** | **11.882** | **2.62×** | **$0.001142** | **0.9034** | 0.3636 | **36.400** | **3.14×** | **$0.003497** | **0.8848** | 0.4385 |
| 8 步 + cache `RDT=0.32` | 10.543 | 2.95× | $0.001013 | 0.8836 | 0.3688 | 31.001 | 3.68× | $0.002978 | 0.8606 | 0.4398 |

**768p 全叠之后的单位成本 = 以前 480p/20 步的成本**（$0.002978 vs $0.002991）。
turbo 单独用就已经比之前最好的加速档（20 步 + Cache-DiT，768p $0.006755）再省 33%，叠 R24 省 48%。

推荐 **8 步 + `RDT=0.24`**，不推荐 R32：再快 17%，但 768p SSIM 从 0.8848 掉到 0.8606，
掉出 Cache-DiT 交付档 ~0.92 那个量级。

### 2 卡（Ulysses=2，5 s）

turbo 把每步压薄了，加卡收益**没有**被稀释：

| 档 | 1 卡 | 2 卡 | 加卡 | $/成片秒 1 卡 → 2 卡 |
|---|---|---|---|---|
| 480p stock 20 步 | 31.137 | 25.127 | 1.24× | $0.002991 → $0.004828 |
| 480p turbo 8 步 | 14.581 | 11.242 | 1.30× | $0.001401 → $0.002160 |
| **480p turbo 8 步 + `RDT=0.24`** | **11.882** | **8.983** | **1.32×** | **$0.001142 → $0.001726** |
| 768p stock 20 步 | 114.155 | 77.257 | 1.48× | $0.010967 → $0.014844 |
| 768p turbo 8 步 | 47.315 | 31.502 | 1.50× | $0.004546 → $0.006053 |
| **768p turbo 8 步 + `RDT=0.24`** | **36.400** | **24.020** | **1.52×** | **$0.003497 → $0.004615** |

加卡倍数沿 stock → turbo → turbo+cache **单调微升**（480p 1.24→1.30→1.32、768p 1.48→1.50→1.52）：
turbo 减的是步数、cache 跳的是整块（连那一块的 all-to-all 一起跳），每步的算力:通信比没变。
对比 sage —— 它只压算力不压通信，768p 加卡从 1.62× 掉到 1.50×。

累计：768p 114.155 s → 36.400 s（单卡 3.14×）→ **24.020 s（2 卡 4.75×）**。2 卡买的是延迟，
单价仍然更高，单卡永远是最便宜的一档。

### ref2va（同一份 LoRA 合进 Ref2VA 分区）

**LoRA 库里没有 ref2va 专用权重**（整库只有 t2v 命名的文件），但同一份 `v4_step600_ema` 合进
Ref2VA 分区就能用：merged 259/259、最大 |delta|/|W| = 0.0036、量化 worst rel 0.0950，与 fl2va 同量级。

```bash
docker exec -e SRC=/models/MiniMax-H3/Ref2VA/transformer \
  -e LORA=/out/lora/minimax_h3_turbo_v4_step600_ema.safetensors \
  -e DST=/out/turbo_ref2va_bf16 h3n python3 /tmp/lora_merge_transformer.py
docker exec -e SRC=/out/turbo_ref2va_bf16 -e DST=/out/nvfp4_ref2va_turbo.safetensors \
  h3n python3 /tmp/nvfp4_quantize_transformer.py
```

| 档（参考短边 1024） | 480p 1 卡 | 480p 2 卡 | 768p 1 卡 | 768p 2 卡 |
|---|---|---|---|---|
| stock NVFP4 20 步 | 36.489 | 28.814 | 118.836 | 79.317 |
| turbo 8 步 | 16.712 | 12.696 | 49.056 | 32.301 |
| **turbo 8 步 + `RDT=0.24`** | **13.513** | **10.077** | **37.727** | **24.651** |
| 对同卡数 stock 20 步 | 2.18× / **2.70×** | 2.27× / 2.86× | 2.42× / **3.15×** | 2.46× / **3.22×** |
| $/成片秒（8 步 + `RDT=0.24`） | **$0.001298** | $0.001936 | **$0.003624** | $0.004736 |

倍数比 fl2va 还略高（fl2va 2.62× / 3.14×），因为参考图编码那份固定开销不随步数缩；绝对时间比
fl2va 贵 3.6%（768p）–14%（480p）。同轮 stock 20 步与顶部交付表逐格吻合（0.2% 内）。

**ref2va 的 SSIM 不能当画质判据**：只给主体图、不给首帧，构图和运镜本身就允许不同，
motion 能量 0.78–1.36（fl2va 只有 0.34–0.52），运动一大、逐像素相关度就被打穿 ——
对 stock 20 步只有 0.59–0.86。目视四帧对照（stock 20 步 / turbo 8 步 / turbo 8 步+R24 三行）
是同构图、同光照、猫的品种与花纹一致、毛发细节一致，无 artifact、无糊、无闪烁。

### 片长曲线（attention 的平方项到底占多少）

attention 是**无 mask 的 packed full self-attention**（视频 + 音频 token 全打在一起），
所以那一项确实随片长平方涨；但 QKV/O 投影和 SwiGLU FFN（28672）是线性的，固定开销还会随片长摊薄。
帧数按 17n+5 对齐、24 fps：5.0 s→124 帧 / 10.0 s→243 帧 / 15.0 s→362 帧（比值 1 : 1.960 : 2.919）。

turbo 8 步载体，`+ cache RDT=0.24` 是交付主力档：

| | 480p 1 卡 | 480p 2 卡 | 加卡 | 768p 1 卡 | 768p 2 卡 | 加卡 |
|---|---|---|---|---|---|---|
| **base** 5 s | 14.581 | 11.242 | 1.30× | 47.315 | 31.502 | 1.50× |
| **base** 10 s | 34.172 | 23.621 | 1.45× | 132.419 | 79.045 | 1.68× |
| **base** 15 s | 61.240 | 39.734 | 1.54× | 255.412 | 147.657 | 1.73× |
| **+cache** 5 s | **11.882** | 8.983 | 1.32× | **36.400** | 24.020 | 1.52× |
| **+cache** 10 s | **27.086** | 18.497 | 1.46× | **99.869** | 59.320 | 1.68× |
| **+cache** 15 s | **47.698** | 30.753 | 1.55× | **190.645** | 109.758 | 1.74× |

$/成片秒（交付主力档）：480p 1 卡 $0.001142 / $0.001330 / $0.001572（5/10/15 s），2 卡 $0.001726 /
$0.001817 / $0.002027；768p 1 卡 $0.003497 / $0.004904 / $0.006284，2 卡 $0.004615 / $0.005826 /
$0.007236。

1. **超线性，但远不到平方。** 按帧数比取指数：480p 单卡 n^1.27（10 s）/ n^1.34（15 s），
   768p 单卡 n^1.53 / n^1.57。纯 O(n²) 时 15 s 该是 5 s 的 8.52×，实测 480p 4.20× / 768p 5.40×。
   768p 指数更高，因为 token 更多、平方项占比更大。
2. **加卡是长片的解，而且越长越划算。** Ulysses 切的正是序列轴，2 卡时每卡的平方项降到 1/4，
   加卡倍数随片长单调上升：480p 1.30→1.45→1.54，768p 1.50→1.68→**1.73（效率 87%）**；
   等价说法是 2 卡把指数按下去：768p n^1.57 → **n^1.44**。零代码改动。
3. **Cache-DiT 在长片上更值钱**（不是被稀释）：480p 1.23×→1.26×→1.28×，768p 1.30×→1.33×→1.34×。
4. **15 s / 768p 单卡不 OOM**：52.6 GB（base）/ 55.5 GB（+cache，多存 residual），2 卡峰值 64.3–64.7 GB。
5. 长片上 cache 的**增量损失不恶化**：对同片长同卡数的 turbo base，8 格 SSIM Y 全在 0.89–0.93
   （768p/15 s 最低 0.8908），motion 一律被 cache 压低一点，与 5 s 的机制一致。

要真正改掉那个指数只能换 sparse/block attention 或者分段生成（便宜但有接缝与漂移风险）。
前者已实测，见下面「稀疏 attention」一节：能把指数从 n^1.57 按到 n^1.43，但有损。

### 画质怎么读（三条）

1. **SSIM 0.88–0.90 是"另一条轨迹"不是"差 10%"。** 同口径对照：stock 自己减到 10 步是
   480p 0.9141 / 768p 0.8999，turbo 8 步与它同量级但**快 19% / 23%**（17.375→14.581、
   58.281→47.315 s）。
2. **没有蒸馏塌运动**：turbo 的运动能量一律 ≥ stock（480p 0.3558 vs 0.3439、768p 0.5193 vs 0.4372）。
   蒸馏失败的签名恰恰是运动塌掉、SSIM 反升。抽四帧看 768p 那条，是同一个连贯运镜、幅度略大，
   没有闪烁。
3. **叠 cache 反而把 SSIM 拉回来一点**（480p 0.9027→0.9034、768p 0.8839→0.8848）：cache 复用早期
   residual = 往前面的轨迹拉，把 turbo 冲过头的运动收回到 stock 附近（0.5193→0.4385）。
   所以"更快 + 画质微升"不矛盾。

### Cache-DiT 的阈值必须重调（这是本轮唯一的坑）

stock 上定的档（480p `RDT=0.20` / 768p `0.16`）在 8 步上**整个偏低**：

- `RDT=0.16` 在 8 步下**一次都不触发**，输出与 base 逐格相同（SSIM 1.000000），把 warmup 从默认 4
  收到 2 也不救。步数少 → 每步 residual 变化大 → 阈值要往上走。turbo 8 步的膝点是 **0.24**。
- **warmup 一定要一起调**：默认 `W=4` 意味着 8 步里头 4 步必算，只剩 4 步可跳，再叠 `MC=3`
  限制连跳 ⇒ 默认档等于半残。这轮一律 `SGLANG_CACHE_DIT_WARMUP=2`（连 `_SECONDARY_WARMUP`）。
- `RDT` 的平台效应照旧：480p 上 `0.28` 与 `0.24` 输出**逐位相同**，别报成两档。

```bash
SGLANG_CACHE_DIT_ENABLED=1 \
SGLANG_CACHE_DIT_WARMUP=2 SGLANG_CACHE_DIT_SECONDARY_WARMUP=2 \
SGLANG_CACHE_DIT_RDT=0.24 SGLANG_CACHE_DIT_SECONDARY_RDT=0.24
```

### 复现性

turbo 8 步 base 跨 3 个 arm、跨 server 重启量了 3 次：480p 14.597 / 14.575 / 14.571（散布 0.18%）、
768p 47.400 / 47.264 / 47.280（0.29%），三份 mp4 的 md5 完全相同 ⇒ 单卡逐位可复现在 turbo 权重上
照旧成立（SSIM 地板 = 1.0）。同轮 stock 20 步与上面交付表差 0.15–0.3%，两张表可并列。

跑法：`PHASES="tb0 tbo tbc tbc2" ./g7e_turbo.sh`（四个 phase 一个 TAG，别共用 TAG，同名 mp4 会静默覆盖）。

## 稀疏 attention（sol_attn）——长片的可选加速档

一句话：**能用、收益随片长上升、单独用被 Cache-DiT 支配、但两个能叠。** 15 s / 768p 单卡从交付
主力的 190.645 s 压到 **148.561 s（对 sage 基线 1.720×，$/成片秒 −22%）**，代价 SSIM 0.818。

### 能挂到 H3 上的稀疏后端只有 2 个

判据还是那条 `packed_varlen`（后端类有没有覆写 `forward_varlen`，见上一节）。c0b6474 的 20 个
后端里实现了它的只有 6 个，稀疏的只有 2 个：

| 后端 | 能挂 | 状态 |
|---|---|---|
| `subblock_sparse_attn` | ✓ | **上游专门在 MiniMax-H3 上调过参**（docstring 就是 H3 t2va 37.7k token：sparsity 0.75→1.14× / 0.85→1.21×，并明确写了会饱和），但 blk64 kernel 只编 `sm_100a`、resolver 的 `required_capability = (10, 0)` 是**严格相等** ⇒ g7e(12.0) 与 B300(10.3) 都开不了，只有 B200/GB200。 |
| `sol_attn` | ✓ | NVlabs/Sana @ `sol-engine` 的 `techniques/sparse_backends`，本节测的这个。 |
| VSA / VMoBA / STA / SVG2(SAP) / block-sparse(Laser) / RainFusion / SLA | ✗ | 缺 `forward_varlen`。**补它对 H3 很便宜**：每次只有一条 packed 序列 ⇒ `q.unsqueeze(0)` 再调自己的 forward，约 15 行。 |

### 为什么"有优化但达不到线性"（源码 + 微基准双证）

`sol_attn/triton_ref/preprocess.py:_diag_threshold_kernel` 最后一行是
`threshold = mean + TAU * std` —— **tau 是 z-score，不是 top-K 预算**。z-score 固定 ⇒ 超阈值的
KV 块**比例**与序列长度无关 ⇒ 保留块数 ∝ n ⇒ attention **仍是 O(n²)**，只是常数被除小。

`scripts/sol_attn_micro.py`（不用起 server，H3 DiT 形状 56 头 / dim 128 / bf16 / 一条 packed 序列）：

| 后端 | 5 s (39760 tok) | 15 s (116060) | 15s/5s | 隐含指数 | 保留块 5s → 15s |
|---|---|---|---|---|---|
| sageattn（交付基线） | 67.66 ms | 566.29 ms | 8.37× | n^1.98 | 密集 |
| sol tau=1.0（默认） | 23.66 | 198.01 | 8.37× | n^1.98 | 15.1% → **15.1%** |
| sol tau=1.5 | 12.34 | 96.94 | 7.86× | n^1.92 | 6.1% → **6.1%** |
| sol tau=2.0 | 7.30 | 49.74 | 6.81× | n^1.79 | 1.9% → **1.9%** |

**保留比例在两个片长上一位不差地相同，对 sage 的加速比也一样（2.859× vs 2.860×）** —— 它把 y 轴
按比例压下来，没动斜率。（随机高斯 q/k 的尾比真实激活轻，所以这是乐观上限，E2E 只拿到 ~77%。）

第二层原因是 Amdahl：用上面片长曲线的三个 768p 单卡点拟合 `T = 20.57n² + 27.80n`（n=1 即 5 s，
常数项 ≈ 0）⇒ 平方项占比 5 s **43.5%** / 10 s 59.7% / 15 s **68.6%**。**5 s 上 attention 全免费
也只值 1.77×——稀疏 attention 是长片的工具。**

### E2E（768p 单卡 turbo 8 步，`dense_steps=2`，参考 = 同片长 sage base）

| 臂 | 5 s | 对 sage | SSIM Y | 15 s | 对 sage | SSIM Y |
|---|---|---|---|---|---|---|
| sage（交付基线） | 47.231 | 1.00× | — | 255.56 | 1.00× | — |
| sol tau=1.0 | 37.909 | 1.246× | 0.8762 | 185.646 | **1.377×** | 0.8082 |
| sol tau=1.5 | 35.178 | 1.343× | 0.8606 | 163.145 | **1.567×** | 0.7682 |
| Cache-DiT R24（交付主力，对照） | 36.400 | 1.300× | — | 190.645 | 1.340× | 0.8928 |

### 叠起来用（sol + Cache-DiT，参考 = **同片长的交付主力那条片**）

| 片长 | sage + Cache-DiT R24 | sol tau=1.0 + Cache-DiT R24 | 加速 | SSIM Y | 平方项占比 |
|---|---|---|---|---|---|
| 5 s | 36.400 | 30.788 | 1.182× | 0.8983 | 43.5% |
| 10 s | 99.869 | 79.994 | 1.248× | 0.8626 | 59.7% |
| 15 s | 190.645 | **148.561** | **1.283×** | 0.8182 | 68.6% |

加速随片长单调上升、SSIM 单调下降，和上面 Amdahl 的平方项占比同向 —— 这是**同一条曲线的两端**，
不是两个独立结论。

### 该用 sage 还是 sol

**默认永远是 sage + Cache-DiT，全片长通用。** sol 是 15 s 长片上「愿意让一档画质换 22% 成本」时
才动的开关，三条硬规则：

1. **sol 和 sage 是二选一**（`--attention-backend` 是全局的，不能叠），所以问题永远是"这一档要不要
   把 sage 换掉"，不是"要不要加 sol"。
2. **永远不要单用 sol。** 10 s 单用 sol 99.52 s vs 单用 Cache-DiT 99.869 s —— **等时间**，但 SSIM
   0.8432 vs 0.8928，等时间更差画质。5 s 上更是又慢又差。
3. **≤5 s 不要动 sage。** 平方项只占 43.5%，1.182× 换 SSIM 0.898 不值；而且 `dense_steps` 默认 10
   意味着 8 步配置下不显式调到 2，sol 一次都不触发（白装）。

15 s 换 sol 的账：190.645 s / SSIM 0.8928 → 148.561 s / SSIM 0.8182，$/成片秒 **$0.006284 →
$0.004897（−22%）**。0.893→0.818 是**肉眼可辨**的细节软化，所以这是商业决定不是技术默认值。

1. **确实能把指数按下去**：sage n^1.576 → tau=1.0 **n^1.483** → tau=1.5 **n^1.432**。参照物是
   加第二张卡（n^1.44）—— tau=1.5 ≈ 白送一张卡的斜率，但**有损**，加卡是无损的。
2. **单独用被 Cache-DiT 严格支配**：15 s 上几乎同加速（1.377× vs 1.340×）而 SSIM 差 **0.085**；
   10 s 上等时间更差画质；5 s 上又慢又差。
3. **画质随片长恶化**（同 tau，0.8762@5s → 0.8082@15s），Cache-DiT 不会（8 格全 0.89–0.93）。
4. **两个能叠**：1.720× = 两者单独之积 1.845 的 **93%**（轻微稀释，cache 跳掉的整块本来也会被
   sol 加速）。
5. 目视五行（sage / sol tau1.0 / tau1.5 / cache / sol+cache，抽第 24/120/240/340 帧）**同构图、
   同运镜、无块状 artifact、无马赛克**；掉分读作"轨迹 + 对比度/色温漂移"，与 turbo vs stock 同性质。

### 四个旋钮 + 两个装它的坑

全走 `--attention-backend-config`（JSON / 文件 / `k=v` 都行；`dense_layers` 带逗号，用 `k=v` 时
写成范围 `0-1` 避开分隔符）。

1. **`dense_steps` 默认 10** —— 前 10 个去噪步强制密集，**跑 ≤10 步（turbo/蒸馏）时 sol 一次都不
   触发**，20 步也只有一半在省。和 Cache-DiT 的 `warmup=4` 是同一个坑。8 步设 2。
2. **`tau`** 是主旋钮（默认 1.0 ≈ 保留 15%）。
3. **`sink_tokens` / `sink_start` 默认 0** —— H3 把文本 token 打进同一条无 mask 的 packed 序列，
   默认配置下**文本 token 也会被剪**；钉成 exact sink 是免费的画质保险。
4. **`kv_splits` 是陷阱**：`interface.py:_validate_cute` 对 `arch != (9,0)` 且 `kv_splits != 1`
   直接 raise，只有 H100 且 seq ≥ 65536 才自动开 4（所以 H100 上长片相对更省）。

- `pip install --no-deps git+https://github.com/NVlabs/Sana.git@sol-engine#subdirectory=techniques/sparse_backends`
  —— **必须 `--no-deps`**：它声明 torch≥2.10，不加会把镜像里的 torch 换掉。cutlass-dsl 镜像已有
  （4.6.2），sm_120 走 `cute_sm120`。
- **c0b6474（本库钉的那版）上开箱即死**：`backends/sol_attn.py` 顶部无条件
  `from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func`，而镜像里
  `flash_attn` 是**空 namespace package**（装的是 flash-attn-4）。导入惰性 ⇒ server 起得来、校验
  也过，第一次走 dense 路径才炸（`Server warmup failed: cannot import name ...`），而 dense 是
  **默认就走的**。修法 `scripts/patches/patch_sol_attn_dense_sage.py`（dense 回退换成 sage，
  顺带让两条路径的密集参考是同一个 kernel）。
  **上游已自行修掉**：`63d783bbe0`（PR #34581，2026-08-18，在 c0b6474 之后）给 sol_attn 加了
  `dense_backend=sage_attn` 与 `_dense_sage()`。换到该 commit 之后的 base 就**不要再打这个补丁**，
  改用 `--attention-backend-config dense_backend=sage_attn`。

```bash
# 微基准
docker cp scripts/sol_attn_micro.py h3n:/tmp/ && docker exec h3n python3 /tmp/sol_attn_micro.py
# E2E：arm 名 solT<tau×10>D<dense_steps>
TAG=sol15 DUR=15.0 ARMS="base_1 solT10D2_1 solT15D2_1" CASES="768_8" \
  CKPT=/out/nvfp4_fl2va_turbo.safetensors ./g7e_dev_levers.sh
# 叠 Cache-DiT（cache 走 env，不用新 knob）
TAG=sol15c DUR=15.0 ARMS="solT10D2_1" CASES="768_8" CKPT=/out/nvfp4_fl2va_turbo.safetensors \
  ENVX_EXTRA="SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_WARMUP=2 \
              SGLANG_CACHE_DIT_SECONDARY_WARMUP=2 SGLANG_CACHE_DIT_RDT=0.24 \
              SGLANG_CACHE_DIT_SECONDARY_RDT=0.24" ./g7e_dev_levers.sh
```

### 真想动阶数，只有两条路

- **把固定比例改成固定预算**：让 tau 跟片长涨，分数近似高斯时保住块数恒定需要
  `tau(n) ≈ tau₀ + √(2·ln(n/n₀))`。纯配置、不碰 kernel；但会把"画质随片长恶化"进一步推高。
- **给一个真·线性的后端补 `forward_varlen`**：STA 是固定 3D 窗口 = O(n·w)，VSA 是固定 top-k 块，
  两个都真降阶，对 H3 各约 15 行，也能顺手给上游发 PR。

**无损的那条已经在手上**：Ulysses 切的正是序列轴，加卡把 768p 从 n^1.576 按到 n^1.44，越长越划算。

## sglang 的并行旋钮（源码核对 + 实测）

我们跑的配置里这些默认全是关的（从 serve 日志的 `server_args` 读）：
`enable_breakable_cuda_graph=false`、`enable_torch_compile=false`、`batching_max_size=1`、
`dp_size=1`、`attention_backend=null`。

| 旋钮 | 结论 |
|---|---|
| `--batching-max-size N` | 原生 diffusion batching 是真实现的（`managers/scheduler.py` + `managers/dynamic_batch_admission.py`，默认 1 = 纯串行），**但 `_can_dynamic_batch()` 里有 `image_path is not None -> return False`** —— fl2va/ref2va 都带输入图，永远进不了 batch，只有纯文本 t2va 能。所以**同一个副本上的并发请求是排队串行的**，别用"单条延迟取倒数"估吞吐。 |
| `--dp-size N` | 内置数据并行：一个 ingress 端口 + 内置负载均衡，可与 `--ulysses-degree` 组合。**这是 g7e.48xlarge 上唯一的吞吐杠杆**，而且一卡只能放一个副本（NVFP4+sage 峰值 52.0 GB，两个副本要 ≤48 GB）。 |
| `--cfg-parallel-size` | **H3 用不了**：`configs/pipeline_configs/minimax_h3.py` 写死 `supports_cfg_parallel=False`，checkpoint 是 CFG 蒸馏的、只有一条正分支。 |
| `--enable-breakable-cuda-graph` | 把 DiT forward 抓成 CUDA graph 段（在 attention 处断开），省 kernel launch 开销、数值无损。**g7e 单卡实测起不来**（`g7e_levers.sh` 的 `bcg_1`）：warmup 阶段 `CUDA error: device-side assert triggered`，从 `layerwise_offload.py:317 prefetch_layer` 的 `copy_stream.wait_stream(...)` / `event.record(stream)` 冒出来，随后 `Server warmup failed; aborting startup` —— 和 g7e 上强制的 text encoder layerwise offload 撞。B300 上 8 卡能起但**白开**（768p 噪声内、480p 慢 3.8%，多吃 35 GB），1 卡也崩（那边是 `illegal memory access`）。结论：H3 的 DiT 不是 launch-bound。 |
| `--enable-torch-compile` | 与上一条互斥，且 flag 自己的帮助文本写了 "will likely cause precision drifts"。 |
| `--attention-backend <后端>` | **H3 的 DiT 走 packed varlen（视频+音频 token 拼成一条不定长序列），所以能选的后端只有 5 个**：`sage_attn`（交付用的 sage 2）、`fa`（sm_120 上被静默降级）、`torch_sdpa`、`sol_attn`、`subblock_sparse_attn`（要 sm_100a，这张卡不行）。判据是 impl 有没有覆写 `forward_varlen`（`backends/attention_backend.py:45`），不满足的在 `minimax_h3.py:192 validate_server_args` 就 `ValueError` 起不来。所以 `sage_attn_3` / `video_sparse_attn` / `vmoba` / `block_sparse_attn` / `sliding_tile_attn` 等**全部对 H3 不可用**，见「被否掉的方案」。稀疏那两个的实测见「稀疏 attention」一节。 |

`serve.sh` 的补丁列表在镜像前进后要收窄（这个循环碰到 `DOES_NOT_APPLY` 是 `exit 1`，不是跳过）。
在 sglang `c0b6474b4` 上 cpu-offload-inplace 已进上游、target-width-height 需要重新 diff：

```bash
PATCHES="minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch" ./serve.sh start
```

多副本共存时**端口按 100 间隔**：sglang 从 `--port` 派生邻居端口（+43 的 HTTP、+44 的 ZMQ
broker），间隔 1 会静默撞死成 hang。`serve.sh` 默认带 `--strict-ports`（否则 sglang 会**静默**
把 HTTP 端口挪走，健康检查就一直等一个没人听的端口）。

## 已知的坑

1. **`--component-attention-backends transformer=sage_attn` 对 H3 的 DiT 完全无效，而且静默**
   ——测出来就是纯 BF16 的数。per-component 后端是只在组件加载期存活的 contextvar，而 H3 的 DiT
   首次 forward 才解析 backend。必须用全局 `--attention-backend sage_attn`。
2. **全局 sage 必须给 text encoder 豁免** `text_encoder=torch_sdpa`：Qwen3VL 的 `LocalAttention`
   只认 `{fa, torch_sdpa}`，否则起服务就死。**在 c0b6474 上要豁免三个**（加 `audio_vae`、
   `video_vae`），见「最新 sglang」一节。已提 issue
   [#35743](https://github.com/sgl-project/sglang/issues/35743)：这几个组件自己声明了
   `default_attention_backend=torch_sdpa`，但 `--attention-backend` 一旦是命令行显式给的，
   selector 就不走 fallback 而是直接 `ValueError` 打死 server。
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
   **两个补丁已提上游**：[#35739](https://github.com/sgl-project/sglang/pull/35739)（sm_120 的
   FP4 GEMM 默认后端 + 无条件 TMA 标度布局，与模型无关）、
   [#35740](https://github.com/sgl-project/sglang/pull/35740)（H3 侧 qkv 逐行标度重排 +
   missing-param 策略）。合进去后这两个补丁和这两个 env 一起退休，退休判据已经验过：
   `FP4_UPSTREAM=1 ./g7e_dev_levers.sh`（跳过补丁、三个 FP4 env 全不设）跑出的成片与打补丁那一路
   **md5 逐位相同**（`174dcfd6…`，768p/8 步/5 s turbo NVFP4）。
10. **g7e 按小时计费，用完就 terminate。** G 系 spot 配额默认只有 64 vCPU，抢不到多卡是配额问题
   不是容量问题；容量在 us-east-1 / eu-central-1 更好。
11. **测 ref2va 时端口必须跟着 variant 走**：`serve.sh` 的 per-variant 默认是 fl2va→30010、
   ref2va→30030（`serve.sh:120`）。拿 30010 去打一个 ref2va 的 server，全程
   `Connection refused`，而 `serve.sh start` 自己是 ready 的、消融脚本也不报 `ARM_FAILED`，
   表现只是整轮 `rc=1` + `inference_time_s` 空 —— 白烧一轮 GPU 才发现。
   `g7e_dev_levers.sh` 现在按 `$VARIANT` 推 `PORT`（prompt 也一样按 variant 推，
   ref2va 那条是「Use \<Picture 1\> as the visual subject…」）。

## 被否掉的方案

- ~~**在 sglang 上跑 NVFP4**~~：**已翻案，现在是交付方案**，见「关于 4-bit」。当时看到的废片
  （SSIM 0.585/0.671、码率 4.2×、运动能量 1.764）是两个 sglang 缺陷叠加，不是 w4a4 这个格式的问题；
  两个补丁在 `scripts/patches/`。留这条是记着教训：**"格式不行"这个结论当时下得太早了**，
  两次撤回（先撤"坏的是 4-bit 激活"，再撤"坏的是 w4a4 本身"）都是因为消融把待测的那条路径绕开了
  （反量化成 bf16 = 不走那个 fp4 GEMM；只比权重误差 = 不过 loader 的行置换）。
- **SageAttention 3**（FP4 attention）：**编得过但 H3 用不了，没有性能数字可报。** 在容器里编译成功
  （`sageattn3-1.0.0`/sm_120a/约 2 分钟，python 3.12 也行），但起服务时
  `configs/pipeline_configs/minimax_h3.py:192 validate_server_args` 用
  `AttentionRequirements(packed_varlen=True)` 去要后端，`sage_attn_3` 没覆写 `forward_varlen`
  → `ValueError: Attention backend 'sage_attn_3' does not implement packed varlen attention`，
  **是报错不是静默回落**（早先写的"装不上会静默降级 TORCH_SDPA"是错的，H3 在参数校验阶段就先拒了）。
  要它能用只能给上游的 `backends/sage_attn3.py` 实现 `forward_varlen`。脚本留在
  `scripts/rejected/build_sageattention3.sh`。同一条判据也封掉了 20 个后端里的 14 个 ——
  **但稀疏后端里有 2 个是过关的**（`sol_attn` 可用、`subblock_sparse_attn` 要 sm_100a），
  见「稀疏 attention」一节。
- **DiT cache / 跳步**：外挂的三套（TeaCache 等）层次上接不上 H3。sglang **自带**的 Cache-DiT 在
  c0b6474 上真挂上了：20 步 1.6×、**30 步 2.1–2.3×**（4 个配置全量过），等成本口径下**优于减步数**
  （30 步 6/6 全赢，最大 +0.032 SSIM），而且是**唯一不稀释加卡收益**的加速手段——**可选加速档**，
  见「最新 sglang」一节。它是有损的，所以不进默认交付配置，但愿意用画质换成本时就开它。
- **FlashAttention-4**：sm_120 两条路全封。
- **裸算力优化**：单卡已在 roofline 上——attention 实测 368.7 TFLOPS（峰值 409.9 的 90%）、四个线性层
  403–414 TFLOPS（100%），trace 里非 matmul 只占 4.1%。所以能动的只有 attention 的**数值精度**
  （= SageAttention），这也是为什么它是唯一有效的杠杆。
