"""
Pre-cache CheXpert Plus PNG images as small JPEGs on a fast drive (SSD/NVMe).

CheXpert Plus PNG files are 3-4 MB each. Loading them from a spinning HDD
is very slow (~24s per batch of 32). This script creates a cache of 256px
(or configurable size) JPEG versions that load ~220× faster from NVMe.

Usage:
    python scripts/cache_chexpert_images.py \\
        --manifest outputs/default/runs/.../manifest.csv \\
        --cache-root /home/user/chexpert256 \\
        --size 256 \\
        --quality 88 \\
        --workers 1

Outputs:
    <cache-root>/patient*/study*/view*.jpg  — resized JPEG files
    <manifest_dir>/manifest_cached256.csv  — manifest with updated image_path

The cached manifest is safe to use directly with cnn-train and cnn-sweep.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from PIL import Image


def cached_path(src: str, cache_root: Path, size: int) -> Path:
    p = Path(src)
    rel = Path(*p.parts[-3:]).with_suffix(".jpg")
    return cache_root / rel


def resize_one(src_path: str, cache_root: Path, size: int, quality: int) -> tuple[str, str | None]:
    dst = cached_path(src_path, cache_root, size)
    if dst.exists():
        return src_path, str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src_path).convert("L")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst, "JPEG", quality=quality)
        return src_path, str(dst)
    except Exception as e:
        print(f"WARN: {src_path}: {e}", flush=True)
        return src_path, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="Path to manifest CSV")
    p.add_argument("--cache-root", required=True, help="Directory to write cached images")
    p.add_argument("--size", type=int, default=256, help="Resize images to this square size (default: 256)")
    p.add_argument("--quality", type=int, default=88, help="JPEG quality (default: 88)")
    p.add_argument("--workers", type=int, default=1,
                   help="Threads for parallel processing (default: 1; HDD benefits from 1, NVMe from 4+)")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    paths = sorted(manifest["image_path"].dropna().unique().tolist())
    print(f"Manifest: {len(manifest)} rows, {len(paths)} unique image paths", flush=True)
    print(f"Cache root: {cache_root} (size={args.size}px, quality={args.quality})", flush=True)

    t0 = time.time()
    path_map: dict[str, str] = {}
    skip = 0

    if args.workers > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(resize_one, p, cache_root, args.size, args.quality): p for p in paths}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                src, dst = fut.result()
                if dst:
                    path_map[src] = dst
                done += 1
                if done % 5000 == 0:
                    elapsed = time.time() - t0
                    eta = (len(paths) - done) / max(done / elapsed, 1)
                    print(f"  {done}/{len(paths)} ({done/len(paths)*100:.1f}%) | ETA {eta/60:.1f}m", flush=True)
    else:
        for i, src_path in enumerate(paths):
            dst = cached_path(src_path, cache_root, args.size)
            if dst.exists():
                path_map[src_path] = str(dst)
                skip += 1
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - t0
                    rate = max((i + 1 - skip) / elapsed, 1)
                    eta = (len(paths) - i - 1) / max(rate + skip / max(elapsed, 1), 1)
                    print(f"  {i+1}/{len(paths)} ({(i+1)/len(paths)*100:.1f}%) "
                          f"| skip={skip} new={i+1-skip} | {rate:.0f} new/s | ETA {eta/60:.1f}m", flush=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                img = Image.open(src_path).convert("L")
                img = img.resize((args.size, args.size), Image.LANCZOS)
                img.save(dst, "JPEG", quality=args.quality)
                path_map[src_path] = str(dst)
            except Exception as e:
                print(f"WARN: {src_path}: {e}", flush=True)

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = max((i + 1 - skip) / elapsed, 1)
                eta = (len(paths) - i - 1) / max(rate + skip / max(elapsed, 1), 1)
                print(f"  {i+1}/{len(paths)} ({(i+1)/len(paths)*100:.1f}%) "
                      f"| skip={skip} new={i+1-skip} | {rate:.0f} new/s | ETA {eta/60:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {len(path_map)}/{len(paths)} images in {elapsed/60:.1f}m", flush=True)

    # Write updated manifest
    manifest_out = manifest_path.parent / f"manifest_cached{args.size}.csv"
    manifest["image_path_cached"] = manifest["image_path"].map(path_map)
    manifest["image_path_orig"] = manifest["image_path"]
    manifest["image_path"] = manifest["image_path_cached"].fillna(manifest["image_path_orig"])
    manifest = manifest.drop(columns=["image_path_cached", "image_path_orig"])
    manifest.to_csv(manifest_out, index=False)
    print(f"Wrote cached manifest: {manifest_out}", flush=True)
    cached_count = manifest["image_path"].str.startswith(str(cache_root)).sum()
    print(f"  {cached_count}/{len(manifest)} rows point to cache ({cached_count/len(manifest)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
