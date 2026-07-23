import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "precache_fastwamhdr_latents_episodewise.py"
_SPEC = importlib.util.spec_from_file_location("precache_fastwamhdr_latents_episodewise", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def main():
    original_args = sys.argv[1:]
    default_args = []
    if "--task" not in original_args and not any(arg.startswith("--task=") for arg in original_args):
        raise SystemExit("--task is required; pass the FastWAM task config name.")
    if "--model" not in original_args and not any(arg.startswith("--model=") for arg in original_args):
        default_args.extend(["--model", "fastwam_joint"])
    if "--data" not in original_args and not any(arg.startswith("--data=") for arg in original_args):
        raise SystemExit("--data is required; pass the FastWAM dataset config name.")
    sys.argv = [sys.argv[0], *default_args, *original_args]
    sys.argv.extend([
        "+data.train.hdr_enabled=false",
    ])
    _MODULE.main()


if __name__ == "__main__":
    main()
