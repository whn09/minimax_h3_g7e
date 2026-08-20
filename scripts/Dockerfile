# H3-on-g7e 交付镜像：把原来散在 serve.sh / build_sageattention.sh / 各 driver 里的 8 处改动
# 一次性烤进镜像。用 ./build_image.sh 建，别直接 docker build（它替你解析 base digest 并做建完自检）。
#
# 为什么值得从"运行时改容器"换成"建镜像"：
#   1. **`:dev` 是移动标签，`h3-validated` 只是本地 re-tag —— 这个保护已经失效过。** 那台 g7e 上
#      两个标签现在指向同一个 ID（c0b6474），也就是说本该durable的"validated"那份被覆盖了。
#      digest 才是真的钉住，而 digest 只有写进 Dockerfile 才有人记得。
#   2. **SageAttention 必须从源码编，而漏掉这一步是静默降级。** pip 上的 wheel 在 sm_120 上只值
#      1.16×（源码是 1.776×），忘了编不会报错，只是慢。（编本身只 2 分钟——`MAX_JOBS=8` 下实测
#      117.6s——所以理由不是"省时间"，是"忘不了"。）
#      这里把"`_qattn_sm89.so` 里必须有 sm_120a cubin"变成了**建镜像时的断言**。
#   3. **两个 NVFP4 补丁也是静默的**：忘打不报错，只出一片灰糊的废片。烤进镜像 = "忘记"不可能。
#   4. 容器重建后要重跑的步骤从 8 步变成 0 步，也不用再记哪个 driver 负责打哪个补丁。
#
# 不烤进去的东西：**权重**（269 GB，照旧 bind mount 到 /models/MiniMax-H3）、量化好的 NVFP4
# checkpoint（24 GB，在 /out）、以及跑起来才有意义的 GPU 数/并行度（那些是 serve.sh 的入参）。
#
# base 默认按 digest 钉死。这个 digest = `lmsysorg/sglang:nightly-dev-20260818-c0b6474b`
# = sglang c0b6474b43363c2f4bc60fe3d7817d393fb51d32（2026-08-18 nightly），torch 2.13.0+cu130,
# python 3.12。要跟 sglang HEAD 就 `BASE=lmsysorg/sglang:dev ./build_image.sh`，但**换 base 就要
# 重新确认 PATCHES 列表**（见下面 ARG PATCHES 的注释）。
ARG BASE=lmsysorg/sglang@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf
FROM ${BASE}

# 在这个 base 上**实际能打**的 .patch。c0b6474 上另外两个不在列表里是有原因的，别加回来：
#   minimax-h3-cpu-offload-inplace.patch  已进上游（decoding.py 里现在是它自己的 rationale 注释）
#   minimax-h3-target-width-height.patch  需要重新 diff（patches/make_patch.sh）
# 换 base 时先用 serve.sh 的 apply 循环试打，再改这里 —— 这条 RUN 是 `git apply` 直接失败即中断，
# 不像 serve.sh 那样有印戳兜底。
ARG PATCHES="minimax-h3-short-edge.patch minimax-h3-mark-missing-params-required.patch"
ARG SAGE_REPO=https://github.com/thu-ml/SageAttention
# d1a57a5 = 2026-08-13 验过的那个 commit，README 所有 sage 数字的出处。
ARG SAGE_REF=d1a57a5
ARG SAGE_ARCH=12.0
# nvcc 并发。建镜像时机器上不该有正在计时的请求，48 vCPU 上 8 是稳的（16 也行，见 build_image.sh）。
ARG JOBS=8
# Sol-Attn（稀疏 attention 消融用，非交付路径）。1 = 装。默认不装：它是从 git 分支装的，
# 会把镜像和一个会动的第三方分支绑在一起，而交付配置用不到它。
ARG WITH_SOL=0

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---- 1) .patch 文件 + 印戳。
# 印戳（/sgl-workspace/.h3-patches/<名字>）是给 serve.sh 看的：它的 apply 循环遇到"打不上"时会去
# 查印戳，有印戳就打印 ALREADY，没有就 `exit 1`（DOES_NOT_APPLY）。烤进镜像的补丁当然打不上第二次，
# 所以没有印戳的话 serve.sh 会在**已经正确**的镜像上直接拒绝启动。
# .h3-image-patches 则是 serve.sh 的 PATCHES 默认值来源，这样"镜像里烤了哪些"只写一遍。
COPY patches/ /patches/
RUN cd /sgl-workspace/sglang \
 && mkdir -p /sgl-workspace/.h3-patches \
 && for n in ${PATCHES}; do \
      test -f "/patches/$n" || { echo "MISSING $n" >&2; exit 1; }; \
      git apply -p1 "/patches/$n"; \
      touch "/sgl-workspace/.h3-patches/$n"; \
      echo "BAKED         $n"; \
    done \
 && printf '%s\n' "${PATCHES}" > /sgl-workspace/.h3-image-patches \
 && bash /patches/inplace_ref_short_edge.sh

# ---- 2) 三个 python 补丁（都是幂等脚本，重复跑打印 ALREADY）。
# 前两个是 NVFP4 必需的 sglang 缺陷修复（TMA 标度布局 / qkv 块标度行序），第三个只在用 sol_attn
# 后端时才走到（把它的 dense 回退从缺失的 flash-attn 换成 sage），对其它配置是死代码。
# 三个都已在上游有对应改动，base 前进过这条线就该逐个删：前两个是 PR #35739 / #35740（未合），
# 第三个已经合了（#34581 / 63d783bbe0，比这个 base 新一天）——换 base 到那之后就删第三行，
# 并改用 `--attention-backend-config dense_backend=sage_attn`。
RUN python3 /patches/patch_nvfp4_tma_scale_layout.py \
 && python3 /patches/patch_h3_qkv_scale_reorder.py \
 && python3 /patches/patch_sol_attn_dense_sage.py \
 && python3 -m compileall -q /sgl-workspace/sglang/python/sglang/multimodal_gen >/dev/null

# ---- 3) SageAttention 从源码编，并断言真的产出了这张卡的 cubin。
# 关键是 TORCH_CUDA_ARCH_LIST：setup.py 里 "Prefer TORCH_CUDA_ARCH_LIST if explicitly specified
# (works without GPUs)"，所以**建镜像不需要 GPU**（docker build 默认也没有 GPU）。
# 2.2.0 在 sm_120 上没有独立的 _qattn_sm120 扩展，它复用 sm89 那套源码、只是多编一份 sm_120a
# cubin —— 所以自检查的是 `_qattn_sm89*.so` 里的 arch 列表，不是有没有某个文件。
# 这个断言不是形式主义：pip wheel 装出来的**文件名一模一样**，差别只在里面没有 sm_120a cubin，
# 运行时静默回落 Triton（1.16× 而不是 1.776×）。
RUN cd /sgl-workspace \
 && git clone --filter=blob:none "${SAGE_REPO}" SageAttention \
 && cd SageAttention \
 && git checkout "${SAGE_REF}" \
 && TORCH_CUDA_ARCH_LIST="${SAGE_ARCH}" MAX_JOBS="${JOBS}" EXT_PARALLEL="${JOBS}" \
    NVCC_APPEND_FLAGS='--threads 2' pip install --no-build-isolation . \
 && rm -rf build \
 && cd /tmp \
 && sagedir=$(python3 -c 'import sageattention,os;print(os.path.dirname(sageattention.__file__))') \
 && want="sm_$(echo "${SAGE_ARCH}" | tr -d .)" \
 && /usr/local/cuda/bin/cuobjdump --list-elf "$sagedir"/_qattn_sm89*.so | grep -q "$want" \
 && echo "SAGE OK       ${SAGE_REF} 带 $want cubin"

# ---- 4) Sol-Attn（可选）。**必须 --no-deps**：它声明 torch>=2.10，不加会把镜像里的 torch 换掉。
# cutlass-dsl 镜像里已有（4.6.2），sm_120 走 cute_sm120 后端。
RUN if [ "${WITH_SOL}" = "1" ]; then \
      pip install --no-deps \
        "git+https://github.com/NVlabs/Sana.git@sol-engine#subdirectory=techniques/sparse_backends" \
      && cd /tmp && python3 -c 'from sol_attn import get_sol_attn_backend; print("SOL OK", get_sol_attn_backend())'; \
    else echo "SOL SKIPPED   （要装：--build-arg WITH_SOL=1）"; fi

# ---- 5) 三个不能忘的 env，直接进镜像。
# RUNAI streamer 会把整个 DiT（62 GiB）攒在匿名内存里，128 GB 的机器上和 CPU-offload 的 text
# encoder 一起把主机推进 direct-reclaim（100% sy）—— 关掉走 mmap。这条对所有配置都必须。
# 另两个只在 NVFP4 路径有意义（对 bf16/fp8 是死开关），但它们和上面第 2 步的补丁是配对的：
# 补丁烤进来了，开关就该跟着进来，否则又回到"两个静默条件必须同时满足"的老坑。
ENV SGLANG_USE_RUNAI_MODEL_STREAMER=0 \
    H3_FP4_TMA_SCALES=1 \
    H3_FP4_QKV_FIX=0

LABEL h3.base="${BASE}" \
      h3.patches="${PATCHES}" \
      h3.sage.ref="${SAGE_REF}" \
      h3.sage.arch="${SAGE_ARCH}" \
      h3.sol="${WITH_SOL}" \
      h3.readme="https://github.com/whn09/minimax_h3_g7e"

# serve.sh 建容器时显式传 `sleep infinity`，所以这只是让 `docker run` 手动起容器时行为一致。
CMD ["sleep", "infinity"]
