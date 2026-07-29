#!/usr/bin/env bash
set -euo pipefail

ROOT="${LIGHTEWM_EVAL_ROOT:-/mnt/data/zhangyu/robot-eval}"
RUNTIME_ROOT="${LIGHTEWM_RUNTIME_ROOT:-${ROOT}}"
LIGHTEWM_ROOT="${LIGHTEWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMPONENT="${1:-all}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export UV_DEFAULT_INDEX
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export GIT_LFS_SKIP_SMUDGE="${GIT_LFS_SKIP_SMUDGE:-1}"

ROBOTWIN_ROOT="${ROOT}/repos/RoboTwin"
ROBOLAB_ROOT="${ROOT}/repos/RoboLab"
STARWAM_ROOT="${ROOT}/repos/StarWAM"
OPENPI_ROOT="${ROOT}/repos/openpi"
ENV_ROOT="${RUNTIME_ROOT}/envs"
CACHE_ROOT="${RUNTIME_ROOT}/cache"
HOST_PYTHON="${LIGHTEWM_HOST_PYTHON:-/usr/bin/python3}"

run_component() {
  local name="$1"
  shift
  if [[ "${COMPONENT}" == "all" || "${COMPONENT}" == "${name}" ]]; then
    "$@"
  fi
}

install_robotwin() {
  local env="${ENV_ROOT}/robotwin-sim"
  local wheelhouse=()
  if [[ -d "${ROOT}/wheelhouse/robotwin" ]]; then
    wheelhouse=(--find-links "${ROOT}/wheelhouse/robotwin")
  fi
  export UV_CACHE_DIR="${CACHE_ROOT}/uv-robotwin"
  uv venv --python 3.10 --allow-existing "${env}"
  uv pip install --python "${env}/bin/python" \
    -r "${ROBOTWIN_ROOT}/script/requirements.txt" \
    "${wheelhouse[@]}"
  local pytorch3d_source="git+https://github.com/facebookresearch/pytorch3d.git@stable"
  if [[ -d "${ROOT}/git-mirrors/pytorch3d/.git" ]]; then
    pytorch3d_source="${ROOT}/git-mirrors/pytorch3d"
  fi
  if ! "${env}/bin/python" -c "import pytorch3d" >/dev/null 2>&1; then
    uv pip install --python "${env}/bin/python" \
      "${pytorch3d_source}" \
      --no-build-isolation
  fi

  local site
  site="$("${env}/bin/python" -c \
    'import site; print(next(path for path in site.getsitepackages() if path.endswith("site-packages")))')"
  sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
    "${site}/sapien/wrapper/urdf_loader.py"
  sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    "${site}/mplib/planner.py"

  if compgen -G "${ROOT}/wheelhouse/robotwin/warp_lang-1.12.0-*.whl" >/dev/null; then
    uv pip install --python "${env}/bin/python" \
      warp-lang==1.12.0 --no-index \
      --find-links "${ROOT}/wheelhouse/robotwin"
  else
    uv pip install --python "${env}/bin/python" warp-lang==1.12.0
  fi
  uv pip install --python "${env}/bin/python" setuptools==69.5.1

  test -f "${ROBOTWIN_ROOT}/envs/curobo/setup.py"
  if ! "${env}/bin/python" -c "import curobo" >/dev/null 2>&1; then
    uv pip install --python "${env}/bin/python" \
      -e "${ROBOTWIN_ROOT}/envs/curobo" --no-build-isolation \
      "${wheelhouse[@]}"
  fi
}

install_starwam() {
  local env="${ENV_ROOT}/lightewm-policy"
  local wheelhouse=()
  if [[ -d "${ROOT}/wheelhouse/starwam" ]]; then
    wheelhouse=(--find-links "${ROOT}/wheelhouse/starwam")
  fi
  export UV_CACHE_DIR="${CACHE_ROOT}/uv-policy"
  uv venv --python 3.11 --allow-existing "${env}"
  uv pip install --python "${env}/bin/python" \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    "${wheelhouse[@]}"
  uv pip install --python "${env}/bin/python" \
    -e "${STARWAM_ROOT}" \
    numpy==1.26.4 safetensors==0.6.2 transformers==4.57.3 \
    diffusers==0.36.0 huggingface-hub==0.36.0 sentencepiece==0.2.1 \
    tokenizers==0.22.1 protobuf==5.29.6 Pillow==11.3.0 \
    opencv-python==4.11.0.86 imageio==2.37.0 imageio-ffmpeg==0.6.0 \
    einops==0.8.1 omegaconf==2.3.0 rich==14.2.0 tqdm==4.67.1 \
    av==12.3.0 pyarrow==22.0.0 \
    "${wheelhouse[@]}"
}

install_robolab() {
  export UV_CACHE_DIR="${CACHE_ROOT}/uv-robolab"
  git -C "${ROBOLAB_ROOT}" lfs pull --include="$(
    printf '%s' \
      'assets/fixtures/Props/instaceable_meshes.usd,' \
      'assets/objects/ycb/textures/obj_000010.png,' \
      'assets/objects/ycb/textures/obj_000013.png,' \
      'assets/objects/hot3d/textures/obj_000030.png'
  )"
  (
    cd "${ROBOLAB_ROOT}"
    printf '%s\n' \
      "b7827de44e5051d8d7803507b9362dbf8521181996449a88934b59ceb0d54db6  assets/fixtures/Props/instaceable_meshes.usd" \
      "b68de45c4d92a2f8d3e8da357912b16b972ef37f25b56bcb593d4cf9fdc4ba2d  assets/objects/ycb/textures/obj_000010.png" \
      "20029d22859dd94150b3d969c0b351663b6d7be31d28b90ba15987ee3ba57fdc  assets/objects/ycb/textures/obj_000013.png" \
      "d77642c56ff641e359b77acb24d93f1b6dd14e5f6a90d3c3f85b8a45c337f197  assets/objects/hot3d/textures/obj_000030.png" \
      | sha256sum -c -
  )
  UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/robolab-sim" \
    uv sync --project "${ROBOLAB_ROOT}" --python 3.11 --extra isaac50 --frozen
}

install_openpi() {
  export UV_CACHE_DIR="${CACHE_ROOT}/uv-openpi"
  local wheelhouse="${RUNTIME_ROOT}/wheelhouse/openpi"
  local runtime_mirror_root="${RUNTIME_ROOT}/git-mirrors"
  local source_mirror
  local runtime_mirror
  local revision
  local upstream
  local git_config_count=0
  local git_config_env=()
  mkdir -p "${runtime_mirror_root}"
  for mirror in lerobot dlimp; do
    source_mirror="${ROOT}/git-mirrors/${mirror}"
    runtime_mirror="${runtime_mirror_root}/${mirror}"
    if [[ -d "${source_mirror}/.git" && ! -d "${runtime_mirror}/.git" ]]; then
      cp -a "${source_mirror}" "${runtime_mirror}"
    fi
  done

  for mapping in \
    "lerobot=https://github.com/huggingface/lerobot" \
    "dlimp=https://github.com/kvablack/dlimp"; do
    local mirror="${mapping%%=*}"
    upstream="${mapping#*=}"
    revision="$(
      awk -v key="${mirror}_revision" \
        '$1 == key ":" {gsub(/"/, "", $2); print $2; exit}' \
        "${LIGHTEWM_ROOT}/env/runtime_versions.yaml"
    )"
    runtime_mirror="${runtime_mirror_root}/${mirror}"
    local mirror_is_complete=true
    if [[ -z "${revision}" ]] \
      || [[ ! -d "${runtime_mirror}/.git" ]] \
      || ! git -C "${runtime_mirror}" rev-parse --verify \
        "${revision}^{commit}" >/dev/null 2>&1 \
      || [[ "$(git -C "${runtime_mirror}" config --get remote.origin.promisor || true)" == "true" ]] \
      || git -C "${runtime_mirror}" rev-list --objects --missing=print \
        "${revision}" | grep -q '^?'; then
      mirror_is_complete=false
    fi
    if [[ "${mirror_is_complete}" != true ]]; then
      local replacement="${runtime_mirror}.complete"
      rm -rf "${replacement}"
      git init -q "${replacement}"
      git -C "${replacement}" fetch --depth=1 "${upstream}" "${revision}"
      if git -C "${replacement}" rev-list --objects --missing=print \
        "${revision}" | grep -q '^?'; then
        echo "Incomplete ${mirror} mirror after fetching ${revision}" >&2
        exit 1
      fi
      rm -rf "${runtime_mirror}"
      mv "${replacement}" "${runtime_mirror}"
    fi
  done

  git config --global --unset-all \
    url."file://${ROOT}/git-mirrors/lerobot".insteadOf || true
  git config --global --unset-all \
    url."file://${ROOT}/git-mirrors/dlimp".insteadOf || true
  for mapping in \
    "lerobot=https://github.com/huggingface/lerobot" \
    "dlimp=https://github.com/kvablack/dlimp"; do
    local mirror="${mapping%%=*}"
    local upstream="${mapping#*=}"
    runtime_mirror="${runtime_mirror_root}/${mirror}"
    if [[ -d "${runtime_mirror}/.git" ]]; then
      git_config_env+=(
        "GIT_CONFIG_KEY_${git_config_count}=url.file://${runtime_mirror}.insteadOf"
        "GIT_CONFIG_VALUE_${git_config_count}=${upstream}"
      )
      git_config_count=$((git_config_count + 1))
    fi
  done

  local find_links_args=()
  if [[ "${LIGHTEWM_OPENPI_PREFETCH_WHEELS:-1}" == "1" ]]; then
    mkdir -p "${wheelhouse}" "${RUNTIME_ROOT}/logs"
    uv export \
      --project "${OPENPI_ROOT}" \
      --frozen \
      --no-dev \
      --quiet \
      --format requirements-txt \
      --no-hashes \
      --output-file "${wheelhouse}/resolved.txt"
    "${HOST_PYTHON}" "${LIGHTEWM_ROOT}/scripts/export_uv_lock_wheels.py" \
      --lock "${OPENPI_ROOT}/uv.lock" \
      --requirements "${wheelhouse}/resolved.txt" \
      --output "${wheelhouse}/aria2.txt" \
      --python-tag cp311 \
      --min-size-mib 1 \
      --url-rewrite \
        "https://files.pythonhosted.org=https://mirrors.aliyun.com/pypi"
    aria2c \
      --input-file="${wheelhouse}/aria2.txt" \
      --dir="${wheelhouse}" \
      --max-concurrent-downloads="${LIGHTEWM_OPENPI_DOWNLOAD_CONCURRENCY:-4}" \
      --max-connection-per-server=1 \
      --split=1 \
      --continue=true \
      --file-allocation=none \
      --check-integrity=true \
      --max-tries=20 \
      --retry-wait=5 \
      --connect-timeout=30 \
      --user-agent="pip/25.1" \
      --summary-interval=10 \
      > "${RUNTIME_ROOT}/logs/openpi-wheel-prefetch.log" 2>&1
  fi
  if compgen -G "${wheelhouse}/*.whl" >/dev/null; then
    find_links_args=(--find-links "${wheelhouse}")
  fi

  env \
    "GIT_CONFIG_COUNT=${git_config_count}" \
    "${git_config_env[@]}" \
    UV_PROJECT_ENVIRONMENT="${ENV_ROOT}/openpi-server" \
    uv sync \
      --project "${OPENPI_ROOT}" \
      --python 3.11 \
      --frozen \
      --no-dev \
      "${find_links_args[@]}"
  uv pip install \
    --python "${ENV_ROOT}/robolab-sim/bin/python" \
    -e "${OPENPI_ROOT}/packages/openpi-client"
}

mkdir -p "${ENV_ROOT}" "${CACHE_ROOT}"
run_component robotwin install_robotwin
run_component starwam install_starwam
run_component robolab install_robolab
run_component openpi install_openpi

if [[ "${COMPONENT}" != "all" ]] && \
  [[ "${COMPONENT}" != "robotwin" ]] && \
  [[ "${COMPONENT}" != "starwam" ]] && \
  [[ "${COMPONENT}" != "robolab" ]] && \
  [[ "${COMPONENT}" != "openpi" ]]; then
  echo "Unknown component: ${COMPONENT}" >&2
  exit 2
fi

echo "Closed-loop benchmark environments are ready (${COMPONENT})."
echo "Runtime root: ${RUNTIME_ROOT}"
