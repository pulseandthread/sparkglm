#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Start one rank of the Atlas SparkGLM image without LLooM. Start rank 1 (the
# worker) first, then rank 0 (the leader, which serves the OpenAI API on
# 127.0.0.1:8893). Container flags, mounts and environment match the LLooM
# recipe linux-nvidia-dgx-spark-2x-glm53-atlas; this script only starts the
# prepared image and never builds, downloads or converts anything.
set -euo pipefail
usage() {
  cat >&2 <<'EOF'
usage: start-node.sh --rank 0|1 --leader-address IP --model-root DIR --image TAG
                     [--fabric-interface IFACE] [--fabric-hca rocep1s0f0]
                     [--profile 4x512k|8x128k|FILE] [--gpu-memory-utilization 0.80-0.95]
                     [--prefix-cache-dir DIR [--prefix-cache-gb 16-100]]
                     [--display-carveout] [--cuda-cache DIR] [--name NAME]
                     [--living-dir DIR] [--dump-hidden DIR]
                     [--steer-dir DIR [--steer-layer N] [--steer-scale D]]
                     [--dump-layer-hidden DIR [--dump-layer-rows all]]

  --leader-address   rank 0's IPv4 address on the direct Spark-to-Spark fabric
  --model-root       directory holding nvidia--GLM-5.3-Flash-NVFP4,
                     incoai--GLM-5.3-Flash-DFlash2 and atlas-overlay (convert.sh)
  --image            the image tag, install/build.sh --tag
  --fabric-interface the fabric's network interface on THIS node (default: the
                     interface of --fabric-hca, e.g. enp1s0f0np0)
  --profile          a profile shipped in the image (default 4x512k) or a JSON
                     file of your own; use the same one on both ranks
  --gpu-memory-utilization  share of unified memory for the engine (default:
                     the profile's 0.88); raise it on dedicated hosts for a
                     larger KV pool and prefix cache
  --prefix-cache-dir keep evicted prefix-cache entries in this directory on
                     the node's own disk (ext4/xfs) instead of dropping them
                     (default: off)
  --prefix-cache-gb  their disk budget in GiB (default 48); the same on both ranks
  --display-carveout lend the GB10's 2 GiB display carveout to the KV cache
                     (2 GiB more KV cache). The container starts with
                     CAP_SYS_ADMIN to export it, which the server drops before
                     it loads anything; use it on both ranks (default: off)
  --cuda-cache       persistent CUDA JIT cache (default: ~/.cache/atlas-cuda)
  --living-dir       living-weights sidecar directory (index.json + safetensors),
                     mounted read-only and passed as --living-dir; LIVING_TOPICS
                     in the environment is forwarded; the same on both ranks
  --steer-dir        steering vectors (index.json + safetensors with v), mounted
                     read-only; added after layer N (--steer-layer, default the
                     middle layer) at dose D (--steer-scale, default 0; set it
                     live with POST /v1/living/steer); the same on both ranks
  --dump-layer-hidden append the residual rows after that layer to
                     DIR/hidden.f32 + DIR/index.jsonl (rank 0 writes; the last
                     prompt token, or every token with --dump-layer-rows all)
  --dump-hidden      append the pre-final-norm rows to DIR/hidden.f32 +
                     DIR/index.jsonl (ATLAS_DUMP_LAST_HIDDEN; rank 0 writes)
EOF
  exit 2
}
rank="" leader="" iface="" model_root="" image="" hca=rocep1s0f0
cache="$HOME/.cache/atlas-cuda" name="" profile=4x512k util="" pc_dir="" pc_gb="" carveout=""
living="" dump="" steer="" steer_layer="" steer_scale="" dump_layer="" dump_layer_rows=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rank) rank=$2; shift 2 ;;
    --leader-address) leader=$2; shift 2 ;;
    --fabric-interface) iface=$2; shift 2 ;;
    --fabric-hca) hca=$2; shift 2 ;;
    --model-root) model_root=$2; shift 2 ;;
    --image) image=$2; shift 2 ;;
    --cuda-cache) cache=$2; shift 2 ;;
    --name) name=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --gpu-memory-utilization) util=$2; shift 2 ;;
    --prefix-cache-dir) pc_dir=$2; shift 2 ;;
    --prefix-cache-gb) pc_gb=$2; shift 2 ;;
    --display-carveout) carveout=1; shift ;;
    --living-dir) living=$2; shift 2 ;;
    --dump-hidden) dump=$2; shift 2 ;;
    --steer-dir) steer=$2; shift 2 ;;
    --steer-layer) steer_layer=$2; shift 2 ;;
    --steer-scale) steer_scale=$2; shift 2 ;;
    --dump-layer-hidden) dump_layer=$2; shift 2 ;;
    --dump-layer-rows) dump_layer_rows=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ $rank == 0 || $rank == 1 ]] && [[ -n $leader && -n $model_root && -n $image ]] || usage
if [[ -z $iface ]]; then
  iface=$(ls "/sys/class/infiniband/$hca/device/net" 2>/dev/null | head -1)
  [[ -n $iface ]] || { echo "no network interface for $hca; pass --fabric-interface" >&2; exit 2; }
fi
model_root=$(realpath "$model_root")
overlay="$model_root/atlas-overlay"
checkpoint="$model_root/nvidia--GLM-5.3-Flash-NVFP4"
drafter="$model_root/incoai--GLM-5.3-Flash-DFlash2"
for path in "$checkpoint/config.json" "$drafter/config.json" "$overlay/sparkglm-verified.json"; do
  [[ -f $path ]] || { echo "missing $path" >&2; exit 2; }
done
mkdir -p "$cache"
profile_mount=()
if [[ $profile == */* || $profile == *.json ]] && [[ ! -f $profile ]]; then
  echo "missing profile file $profile" >&2; exit 2
fi
if [[ -f $profile ]]; then
  profile_mount=(--mount "type=bind,src=$(realpath "$profile"),dst=/opt/atlas/profiles/custom.json,readonly")
  profile=custom
fi
if [[ -n $pc_dir || -n $pc_gb ]]; then
  pc_gb=${pc_gb:-48}
  [[ $pc_dir == /* && $pc_dir != *,* ]] || { echo "--prefix-cache-dir must be set to an absolute path without commas" >&2; exit 2; }
  [[ $pc_gb =~ ^[1-9][0-9]{1,2}$ ]] && (( pc_gb >= 16 && pc_gb <= 100 )) ||
    { echo "--prefix-cache-gb must be a whole number from 16 to 100" >&2; exit 2; }
fi
name=${name:-atlas-sparkglm-rank$rank}
docker rm -f "$name" >/dev/null 2>&1 || true
# The engine unlinks its cache files while they are open and sweeps any a crash
# left behind, so nothing carries over a restart.
pc_args=()
if [[ -n $pc_dir ]]; then
  mkdir -p "$pc_dir/kv" "$pc_dir/ssm"
  pc_dir=$(realpath "$pc_dir")
  case $(stat -f -c %T "$pc_dir" 2>/dev/null) in
    tmpfs | ramfs | overlayfs) echo "--prefix-cache-dir $pc_dir is not on a disk" >&2; exit 2 ;;
  esac
  free_kb=$(df -Pk "$pc_dir" | awk 'NR == 2 {print $4}')
  (( free_kb >= pc_gb * 1048576 )) ||
    { echo "--prefix-cache-dir $pc_dir has $((free_kb / 1048576)) GiB free; --prefix-cache-gb needs $pc_gb" >&2; exit 2; }
  pc_args=(--mount "type=bind,src=$pc_dir,dst=/prefix-cache" -e SPARKGLM_PREFIX_CACHE_GB="$pc_gb")
fi
# One process per host may hold the carveout; every container shares this
# host directory's lock file, which the server holds while it runs.
carveout_args=()
if [[ -n $carveout ]]; then
  lock_dir=/run/lock/sparkglm
  mkdir -p "$lock_dir" && chmod 1777 "$lock_dir" 2>/dev/null || true
  carveout_args=(--cap-add SYS_ADMIN --mount "type=bind,src=$lock_dir,dst=$lock_dir"
    -e SPARKGLM_DISPLAY_CARVEOUT=1)
fi
# Living weights: the sidecar directory read-only, the dump directory writable,
# both at their host paths.
living_args=()
if [[ -n $living ]]; then
  [[ -f $living/index.json ]] || { echo "missing $living/index.json" >&2; exit 2; }
  living=$(realpath "$living")
  living_args+=(--mount "type=bind,src=$living,dst=$living,readonly" -e SPARKGLM_LIVING_DIR="$living")
  [[ -z ${LIVING_TOPICS:-} ]] || living_args+=(-e LIVING_TOPICS="$LIVING_TOPICS")
fi
if [[ -n $dump ]]; then
  mkdir -p "$dump" && dump=$(realpath "$dump")
  living_args+=(--mount "type=bind,src=$dump,dst=$dump" -e SPARKGLM_DUMP_LAST_HIDDEN="$dump")
fi
if [[ -n $steer ]]; then
  [[ -f $steer/index.json ]] || { echo "missing $steer/index.json" >&2; exit 2; }
  steer=$(realpath "$steer")
  living_args+=(--mount "type=bind,src=$steer,dst=$steer,readonly" -e SPARKGLM_STEER_DIR="$steer")
fi
[[ -z $steer_layer ]] || living_args+=(-e SPARKGLM_STEER_LAYER="$steer_layer")
[[ -z $steer_scale ]] || living_args+=(-e SPARKGLM_STEER_SCALE="$steer_scale")
if [[ -n $dump_layer ]]; then
  mkdir -p "$dump_layer" && dump_layer=$(realpath "$dump_layer")
  living_args+=(--mount "type=bind,src=$dump_layer,dst=$dump_layer" -e SPARKGLM_DUMP_LAYER_HIDDEN="$dump_layer")
fi
[[ -z $dump_layer_rows ]] || living_args+=(-e SPARKGLM_DUMP_LAYER_ROWS="$dump_layer_rows")
# The overlay links into the original checkpoint, so it is mounted at the same
# absolute path inside the container.
docker run -d --name "$name" --restart no --network host --ipc host \
  --shm-size 32g --memory 114g --gpus all --device /dev/infiniband:/dev/infiniband \
  --cap-add IPC_LOCK --cap-add SYS_NICE --ulimit memlock=-1:-1 \
  --security-opt no-new-privileges=true --stop-timeout 60 \
  --mount "type=bind,src=$checkpoint,dst=$checkpoint,readonly" \
  --mount "type=bind,src=$overlay,dst=$overlay,readonly" \
  --mount "type=bind,src=$drafter,dst=$drafter,readonly" \
  --mount "type=bind,src=$cache,dst=/atlas-cuda-cache" ${profile_mount[@]+"${profile_mount[@]}"} ${pc_args[@]+"${pc_args[@]}"} \
  ${carveout_args[@]+"${carveout_args[@]}"} ${living_args[@]+"${living_args[@]}"} \
  -e NODE_RANK="$rank" -e MASTER_ADDR="$leader" -e MASTER_PORT=29510 \
  -e FABRIC_INTERFACE="$iface" -e FABRIC_HCA="$hca" \
  -e MODEL_PATH="$overlay" -e DRAFTER_PATH="$drafter" \
  -e SERVED_MODEL_NAME=glm-5.3-flash-atlas -e SPARKGLM_PROFILE="$profile" \
  ${util:+-e SPARKGLM_GPU_MEMORY_UTILIZATION="$util"} \
  -e ATLAS_WORLD_SIZE=2 -e ATLAS_TP_SIZE=2 -e ATLAS_EP_SIZE=2 -e ATLAS_CONTEXT_WINDOW=524288 \
  -e NCCL_SOCKET_IFNAME="$iface" -e GLOO_SOCKET_IFNAME="$iface" -e NCCL_IB_HCA="$hca" \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_CROSS_NIC=0 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_RETRY_CNT=7 -e NCCL_IB_TIMEOUT=22 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple -e NCCL_BUFFSIZE=33554432 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_NCHANNELS=1 \
  -e NCCL_DMABUF_ENABLE=0 -e NCCL_DEBUG=WARN -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_CACHE_PATH=/atlas-cuda-cache -e CUDA_CACHE_MAXSIZE=4294967296 \
  "$image"
echo "started $name (rank $rank); follow with: docker logs -f $name"
