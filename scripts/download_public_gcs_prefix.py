#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def list_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    endpoint = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    token = None
    objects: list[dict[str, Any]] = []
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        with urllib.request.urlopen(
            f"{endpoint}?{urllib.parse.urlencode(params)}",
            timeout=60,
        ) as response:
            payload = json.load(response)
        objects.extend(payload.get("items", ()))
        token = payload.get("nextPageToken")
        if not token:
            return objects


def relative_object_path(name: str, prefix: str) -> Path:
    if not name.startswith(prefix):
        raise ValueError(f"object {name!r} is outside prefix {prefix!r}")
    relative = Path(name[len(prefix) :].lstrip("/"))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe object path: {name!r}")
    return relative


def media_url(bucket: str, item: dict[str, Any]) -> str:
    name = urllib.parse.quote(str(item["name"]), safe="")
    query = urllib.parse.urlencode(
        {"alt": "media", "generation": str(item["generation"])}
    )
    return (
        f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/"
        f"{name}?{query}"
    )


def file_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def validate_object(path: Path, item: dict[str, Any]) -> None:
    expected_size = int(item["size"])
    if path.stat().st_size != expected_size:
        raise IOError(
            f"size mismatch for {item['name']}: "
            f"{path.stat().st_size} != {expected_size}"
        )
    expected_md5 = item.get("md5Hash")
    if expected_md5 and file_md5_base64(path) != expected_md5:
        raise IOError(f"MD5 mismatch for {item['name']}")


def download_object(
    bucket: str,
    prefix: str,
    output_dir: Path,
    item: dict[str, Any],
    *,
    retries: int,
) -> dict[str, Any]:
    destination = output_dir / relative_object_path(str(item["name"]), prefix)
    expected_size = int(item["size"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == expected_size:
        validate_object(destination, item)
        return {"name": item["name"], "size": expected_size, "status": "cached"}

    partial = destination.with_name(f"{destination.name}.part")
    for attempt in range(retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(media_url(bucket, item), headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                if offset and response.status != 206:
                    offset = 0
                    mode = "wb"
                else:
                    mode = "ab" if offset else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
            actual_size = partial.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"size mismatch for {item['name']}: "
                    f"{actual_size} != {expected_size}"
                )
            os.replace(partial, destination)
            validate_object(destination, item)
            return {
                "name": item["name"],
                "size": expected_size,
                "status": "downloaded",
            }
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a public Google Cloud Storage prefix with resume support."
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    prefix = args.prefix.rstrip("/") + "/"
    output_dir = Path(args.output_dir)
    objects = list_objects(args.bucket, prefix)
    if not objects:
        raise ValueError(f"no public objects found at gs://{args.bucket}/{prefix}")
    print(
        f"Found {len(objects)} objects "
        f"({sum(int(item['size']) for item in objects)} bytes)",
        flush=True,
    )

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_object,
                args.bucket,
                prefix,
                output_dir,
                item,
                retries=args.retries,
            ): item
            for item in objects
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['status']}: {result['name']} ({result['size']} bytes)",
                flush=True,
            )

    manifest = {
        "schema_version": 1,
        "source": f"gs://{args.bucket}/{prefix}",
        "objects": [
            {
                "name": item["name"],
                "size": int(item["size"]),
                "generation": item["generation"],
                "md5Hash": item.get("md5Hash"),
                "crc32c": item.get("crc32c"),
            }
            for item in sorted(objects, key=lambda value: value["name"])
        ],
        "total_bytes": sum(int(item["size"]) for item in objects),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
