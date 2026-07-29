from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "setup_aliyun_l20_runtime.sh"
ENV_SCRIPT = Path(__file__).parents[1] / "scripts" / "setup_closed_loop_benchmark_envs.sh"


def test_cuda_repository_is_enabled_before_compat_package_install() -> None:
    script = SCRIPT.read_text()
    compat_install = "ensure_cuda_apt_repo\n  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-compat-12-8"

    assert compat_install in script


def test_runtime_installs_pinned_uv_when_missing() -> None:
    script = SCRIPT.read_text()

    assert 'uv_version="${LIGHTEWM_UV_VERSION:-0.11.32}"' in script
    assert '"uv==${uv_version}"' in script


def test_runtime_installs_git_lfs_for_benchmark_assets() -> None:
    script = SCRIPT.read_text()

    assert "git git-lfs" in script
    assert "git lfs install --system" in script


def test_runtime_replaces_the_pai_ppu_cuda_symlink() -> None:
    script = SCRIPT.read_text()

    assert '[[ "$(readlink -f /usr/local/cuda)" == /usr/local/PPU_SDK/* ]]' in script


def test_benchmark_install_does_not_inherit_ppu_cuda_libraries() -> None:
    script = ENV_SCRIPT.read_text()

    assert 'export CUDA_HOME="${LIGHTEWM_CUDA_HOME:-/usr/local/cuda}"' in script
    assert 'sanitize_library_path "${LD_LIBRARY_PATH:-}" "/usr/local/PPU_SDK"' in script


def test_robotwin_installs_wheel_before_curobo_metadata_build() -> None:
    script = ENV_SCRIPT.read_text()

    build_tools = "setuptools==69.5.1 wheel==0.47.0"
    curobo_install = '-e "${ROBOTWIN_ROOT}/envs/curobo" --no-build-isolation'
    assert build_tools in script
    assert script.index(build_tools) < script.index(curobo_install)


def test_openpi_prefetch_is_consumed_by_pip_sync() -> None:
    script = ENV_SCRIPT.read_text()

    assert "uv pip sync" in script
    assert '--python "${ENV_ROOT}/openpi-server/bin/python"' in script
    assert '"${wheelhouse}/resolved.txt"' in script


def test_openpi_installs_its_undeclared_runtime_pytest_dependency() -> None:
    script = ENV_SCRIPT.read_text()

    assert '"pytest==8.3.5"' in script
