#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def wheel_compatibility_score(
    filename: str,
    *,
    python_tag: str,
) -> tuple[int, int, int] | None:
    lower = filename.lower()
    if not lower.endswith(".whl"):
        return None
    if any(
        token in lower
        for token in ("aarch64", "ppc64", "s390x", "win_", "macosx", "musllinux")
    ):
        return None
    if not (lower.endswith("x86_64.whl") or lower.endswith("-any.whl")):
        return None
    tags = lower[:-4].rsplit("-", 3)
    if len(tags) != 4:
        return None
    wheel_python, abi, _platform = tags[-3:]
    if wheel_python in {"py3", "py2.py3"}:
        python_score = 1
    elif wheel_python == python_tag:
        python_score = 3
    elif (
        abi == "abi3"
        and wheel_python.startswith("cp")
        and int(wheel_python[2:]) <= int(python_tag[2:])
    ):
        python_score = 2
    else:
        return None

    manylinux = re.search(r"manylinux_(\d+)_(\d+)_x86_64", lower)
    if manylinux:
        platform_score = (int(manylinux.group(1)), int(manylinux.group(2)))
    elif "manylinux2014_x86_64" in lower:
        platform_score = (2, 17)
    elif "manylinux2010_x86_64" in lower:
        platform_score = (2, 12)
    elif "manylinux1_x86_64" in lower:
        platform_score = (2, 5)
    else:
        platform_score = (0, 0)
    return python_score, *platform_score


def is_compatible_wheel(filename: str, *, python_tag: str) -> bool:
    return wheel_compatibility_score(filename, python_tag=python_tag) is not None


def load_resolved_requirements(path: Path) -> dict[str, list[Requirement]]:
    requirements: dict[str, list[Requirement]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        requirements.setdefault(canonicalize_name(requirement.name), []).append(
            requirement
        )
    return requirements


def rewrite_url(url: str, rewrites: tuple[tuple[str, str], ...]) -> str:
    for source, target in rewrites:
        if url.startswith(source):
            return target + url[len(source) :]
    return url


def export_aria2_manifest(
    lock_path: Path,
    output_path: Path,
    *,
    python_tag: str,
    min_size: int,
    requirements_path: Path | None = None,
    url_rewrites: tuple[tuple[str, str], ...] = (),
) -> tuple[int, int]:
    payload = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    resolved = (
        load_resolved_requirements(requirements_path)
        if requirements_path is not None
        else None
    )
    lines: list[str] = []
    selected_bytes = 0
    selected_count = 0
    for package in payload.get("package", []):
        package_name = canonicalize_name(package["name"])
        package_version = Version(package["version"])
        if resolved is not None and not any(
            package_version in requirement.specifier
            for requirement in resolved.get(package_name, [])
        ):
            continue
        compatible = [
            wheel
            for wheel in package.get("wheels", [])
            if int(wheel.get("size", 0)) >= min_size
            and is_compatible_wheel(
                unquote(Path(urlparse(wheel["url"]).path).name),
                python_tag=python_tag,
            )
        ]
        if not compatible:
            continue
        wheel = max(
            compatible,
            key=lambda item: wheel_compatibility_score(
                unquote(Path(urlparse(item["url"]).path).name),
                python_tag=python_tag,
            ),
        )
        filename = unquote(Path(urlparse(wheel["url"]).path).name)
        algorithm, separator, digest = wheel["hash"].partition(":")
        if not separator or algorithm != "sha256" or len(digest) != hashlib.sha256().digest_size * 2:
            raise ValueError(f"unsupported wheel hash for {package['name']}")
        lines.extend(
            (
                rewrite_url(wheel["url"], url_rewrites),
                f" out={filename}",
                f" checksum=sha-256={digest}",
            )
        )
        selected_count += 1
        selected_bytes += int(wheel["size"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected_count, selected_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pinned Linux wheels from a uv.lock as an aria2 input file."
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--requirements",
        help="Optional uv-exported requirements file used to select exact versions.",
    )
    parser.add_argument(
        "--url-rewrite",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Rewrite a wheel URL prefix while retaining the lockfile hash.",
    )
    parser.add_argument("--python-tag", default="cp311")
    parser.add_argument("--min-size-mib", type=float, default=1.0)
    args = parser.parse_args()
    rewrites = []
    for value in args.url_rewrite:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise ValueError(f"--url-rewrite must be SOURCE=TARGET, got {value!r}")
        rewrites.append((source, target))
    count, size = export_aria2_manifest(
        Path(args.lock),
        Path(args.output),
        python_tag=args.python_tag,
        min_size=int(args.min_size_mib * 1024 * 1024),
        requirements_path=Path(args.requirements) if args.requirements else None,
        url_rewrites=tuple(rewrites),
    )
    print(f"Exported {count} wheels ({size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
