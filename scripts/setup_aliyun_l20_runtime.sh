#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIGHTEWM_EVAL_ROOT:-/mnt/data/zhangyu/robot-eval}"
CACHE_DIR="${ROOT}/cache"
LOG_DIR="${ROOT}/logs"
mkdir -p "${CACHE_DIR}" "${LOG_DIR}"

ensure_cuda_apt_repo() {
  local repo_list="/etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list"
  if [[ ! -s "${repo_list}" ]]; then
    curl -fsSL \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
      -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
  fi
  apt-get update -qq
}

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  aria2 ca-certificates curl git git-lfs libglu1-mesa python3-packaging python3-pip python3-tomli vulkan-tools \
  xserver-xorg-core
git lfs install --system --skip-repo

uv_version="${LIGHTEWM_UV_VERSION:-0.11.32}"
if ! command -v uv >/dev/null 2>&1 \
  || [[ "$(uv --version 2>/dev/null)" != "uv ${uv_version} "* ]]; then
  python3 -m pip install \
    --no-cache-dir \
    --index-url "${LIGHTEWM_PYPI_INDEX:-https://mirrors.aliyun.com/pypi/simple}" \
    "uv==${uv_version}"
fi

need_cuda_packages=false
if [[ ! -x /usr/local/cuda/bin/nvcc ]] \
  || strings /usr/local/cuda/bin/nvcc | grep -q 'PPU_OPTION' \
  || [[ "$(readlink -f /usr/local/cuda)" == /usr/local/PPU_SDK/* ]]; then
  if [[ -L /usr/local/cuda-12.4 ]] \
    && [[ "$(readlink -f /usr/local/cuda-12.4)" == /usr/local/PPU_SDK/CUDA_SDK ]]; then
    mv /usr/local/cuda-12.4 /usr/local/cuda-12.4-hggc
  fi
  need_cuda_packages=true
fi
for header in cublas_v2.h cusolverDn.h cusparse.h; do
  if [[ ! -f "/usr/local/cuda/include/${header}" ]]; then
    need_cuda_packages=true
  fi
done
if [[ "${need_cuda_packages}" == true ]]; then
  ensure_cuda_apt_repo
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    cuda-nvcc-12-4 \
    libcublas-dev-12-4 \
    libcusolver-dev-12-4 \
    libcusparse-dev-12-4
fi
if [[ ! -x /usr/local/cuda/bin/nvcc ]] \
  || strings /usr/local/cuda/bin/nvcc | grep -q 'PPU_OPTION'; then
  echo "A stock NVIDIA CUDA 12.4 compiler is required for RoboTwin/CuRobo." >&2
  exit 1
fi
for header in cublas_v2.h cusolverDn.h cusparse.h; do
  if [[ ! -f "/usr/local/cuda/include/${header}" ]]; then
    echo "The CUDA development header ${header} is required for RoboTwin/CuRobo." >&2
    exit 1
  fi
done

driver_version="$(
  sed -n 's/^NVRM version:.*Kernel Module  \([^ ]*\).*/\1/p' \
    /proc/driver/nvidia/version
)"
if [[ -z "${driver_version}" ]]; then
  echo "Unable to determine the NVIDIA kernel driver version." >&2
  exit 1
fi
driver_major="${driver_version%%.*}"

driver_root="${CACHE_DIR}/nvidia-${driver_version}"
runfile="${driver_root}/NVIDIA-Linux-x86_64-${driver_version}.run"
extracted="${driver_root}/extracted"
runtime_env="${driver_root}/runtime_env.sh"
mkdir -p "${driver_root}"

driver_lib_dir="/usr/lib/x86_64-linux-gnu"
for library in libnvidia-ml libcuda; do
  versioned="${driver_lib_dir}/${library}.so.${driver_version}"
  if [[ -f "${versioned}" && ! -e "${driver_lib_dir}/${library}.so.1" ]]; then
    ln -sfn "$(basename "${versioned}")" "${driver_lib_dir}/${library}.so.1"
    ln -sfn "${library}.so.1" "${driver_lib_dir}/${library}.so"
  fi
done

if [[ ! -e /dev/nvidia-modeset ]] && grep -q '^195 nvidia-modeset$' /proc/devices; then
  mknod -m 666 /dev/nvidia-modeset c 195 254
fi

if ! grep -q '^c 195:254 rw' /sys/fs/cgroup/devices/devices.list; then
  echo "The DSW container cannot access nvidia-modeset." >&2
  echo "Set NVIDIA_DRIVER_CAPABILITIES=all on the DSW instance and restart it." >&2
  exit 1
fi
if ! grep -q '^nvidia_modeset ' /proc/modules; then
  echo "The host has not loaded the nvidia_modeset kernel module." >&2
  exit 1
fi

if [[ ! -f "${extracted}/libGLX_nvidia.so.${driver_version}" ]]; then
  if [[ ! -f "${runfile}" ]]; then
    curl -fL \
      "https://download.nvidia.com/XFree86/Linux-x86_64/${driver_version}/NVIDIA-Linux-x86_64-${driver_version}.run" \
      -o "${runfile}"
    chmod +x "${runfile}"
  fi
  rm -rf "${extracted}"
  sh "${runfile}" --extract-only --target "${extracted}"
fi

ln -sfn "libGLX_nvidia.so.${driver_version}" "${extracted}/libGLX_nvidia.so.0"
ln -sfn "libEGL.so.1.1.0" "${extracted}/libEGL.so.1"
ln -sfn "libEGL.so.1" "${extracted}/libEGL.so"
ln -sfn "libEGL_nvidia.so.${driver_version}" "${extracted}/libEGL_nvidia.so.0"

if [[ -f "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.${driver_version}" ]] \
  && [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  : > "${runtime_env}"
else
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
    "${extracted}" > "${runtime_env}"
  printf 'export VK_ICD_FILENAMES=%q\n' \
    "${extracted}/nvidia_icd.json" >> "${runtime_env}"
fi

# Keep CUDA forward-compat libraries opt-in. Injecting them into Isaac Sim's
# process can hide the host Vulkan driver and make GPU Foundation report
# "Driver Version: 0", even when CUDA compute works without them.
if (( driver_major < 570 )); then
  ensure_cuda_apt_repo
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-12-8
  cuda_compat_dir="/usr/local/cuda-12.8/compat"
  if [[ ! -f "${cuda_compat_dir}/libcuda.so.1" ]]; then
    echo "cuda-compat-12-8 did not provide ${cuda_compat_dir}/libcuda.so.1" >&2
    exit 1
  fi
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' \
    "${cuda_compat_dir}" > "${driver_root}/cuda_compat_env.sh"
fi

xorg_root="${driver_root}/xorg"
xorg_modules="${xorg_root}/modules"
xorg_config="${xorg_root}/xorg.conf"
xorg_log="${LOG_DIR}/Xorg.99.log"
mkdir -p "${xorg_modules}/drivers" "${xorg_modules}/extensions"
ln -sfn "${extracted}/nvidia_drv.so" \
  "${xorg_modules}/drivers/nvidia_drv.so"
ln -sfn "${extracted}/libglxserver_nvidia.so.${driver_version}" \
  "${xorg_modules}/extensions/libglxserver_nvidia.so"

"${extracted}/nvidia-xconfig" \
  --output-xconfig="${xorg_config}" \
  --allow-empty-initial-configuration \
  --silent
sed -i '/^[[:space:]]*Option[[:space:]]*"UseDisplayDevice"/d' "${xorg_config}"
sed -i '/^[[:space:]]*Virtual[[:space:]]/d' "${xorg_config}"
sed -i '/Section "Files"/,/EndSection/d' "${xorg_config}"
sed -i "1i Section \"Files\"\n    ModulePath \"${xorg_modules}\"\n    ModulePath \"/usr/lib/xorg/modules\"\nEndSection" "${xorg_config}"

if ! pgrep -f 'Xorg :99 ' >/dev/null; then
  nohup Xorg :99 \
    -config "${xorg_config}" \
    -noreset \
    -logfile "${xorg_log}" \
    > "${LOG_DIR}/Xorg.99.stdout.log" 2>&1 &
fi
for _ in $(seq 1 20); do
  [[ -S /tmp/.X11-unix/X99 ]] && break
  sleep 1
done
if [[ ! -S /tmp/.X11-unix/X99 ]]; then
  echo "NVIDIA Xorg did not become ready. See ${xorg_log}." >&2
  exit 1
fi

x_runtime_dir="/tmp/lightewm-xdg-runtime"
mkdir -p "${x_runtime_dir}"
chmod 700 "${x_runtime_dir}"
printf 'export DISPLAY=:99\n' >> "${runtime_env}"
printf 'export XDG_RUNTIME_DIR=%q\n' "${x_runtime_dir}" >> "${runtime_env}"
printf 'export OMNI_KIT_ACCEPT_EULA=YES\n' >> "${runtime_env}"

# shellcheck source=/dev/null
source "${runtime_env}"
vulkan_log="${LOG_DIR}/vulkaninfo-${driver_version}.log"
vulkaninfo > "${vulkan_log}" 2>&1
grep -q 'deviceName.*NVIDIA L20' "${vulkan_log}"
grep -q 'VK_KHR_ray_tracing_pipeline' "${vulkan_log}"
grep -q 'VK_KHR_acceleration_structure' "${vulkan_log}"

echo "Vulkan RT runtime ready: ${runtime_env}"
echo "Evidence: ${vulkan_log}"
