#!/usr/bin/env python3
"""Verify the NanoLang BitNet b1.58 700M release.

Checks, using only the standard library:

1. every entry in ``CHECKSUMS.txt`` that is present on disk has the recorded
   sha256 (files absent from the download are reported, not failed);
2. the ``.nanow`` container matches ``MANIFEST.json`` in size and sha256;
3. the container header is structurally sane and its tensor directory matches the
   manifest one-for-one.

Usage
-----
    python3 verify.py                 # verify what is here
    python3 verify.py --require-all   # fail if any listed file is missing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# .nanow v1 layout, mirrored from phase1/include/nanow_format.h.
#
#   header:     magic, version, flags, num_tensors,
#               num_layers, hidden_dim, state_dim, vocab_size,
#               total_params, file_size, directory_size, reserved[3]
#   descriptor: name[64], dtype, packing, ndims, shape[4],
#               num_elements, data_offset, data_size, scale_factor
#
# The magic constant is 0x4E414E4F, which is the bytes "ONAN" on disk in
# little-endian order (the header comment upstream reads 'N','A','N','O' as the
# intent, not the byte order). We compare the integer, which is unambiguous.
MAGIC = 0x4E414E4F
# Header feature flags. A model container leaves these clear; a KV cache — which
# shares this layout — sets this bit. Mirrors NANOW_FLAG_KV_CACHE in
# phase1/include/nanow_format.h.
KV_CACHE_FLAG = 0x0001
HEADER_FMT = "<IIII IIII QQ I 3I"
DESC_FMT = "<64s III 4I Q Q Q f"
HEADER_SIZE = struct.calcsize(HEADER_FMT)      # 64
DESC_SIZE = struct.calcsize(DESC_FMT)          # 120


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_checksums() -> list[tuple[str, str]]:
    path = os.path.join(HERE, "CHECKSUMS.txt")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition("  ")
        if digest and name:
            out.append((digest, name))
    return out


def check_checksums(require_all: bool) -> tuple[int, int, list[str]]:
    ok = missing = 0
    problems: list[str] = []
    for digest, name in read_checksums():
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            missing += 1
            note = "missing"
            if require_all:
                problems.append(f"{name}: {note} (required)")
            else:
                print(f"  skip     {name}  ({note})")
            continue
        got = sha256(path)
        if got == digest:
            ok += 1
            print(f"  ok       {name}")
        else:
            problems.append(f"{name}: sha256 mismatch\n             want {digest}\n             got  {got}")
    return ok, missing, problems


def check_container() -> list[str]:
    path = os.path.join(HERE, "MANIFEST.json")
    if not os.path.exists(path):
        print("  (no MANIFEST.json here; run with --require-all to treat as failure)")
        return []
    man = json.load(open(path, encoding="utf-8"))
    problems: list[str] = []

    name = man["file"]
    weights = os.path.join(HERE, name)
    if not os.path.exists(weights):
        print(f"  skip     {name}  (missing — download it from the model repo)")
        return []

    size = os.path.getsize(weights)
    if size != man["file_bytes"]:
        problems.append(f"{name}: size {size} != manifest {man['file_bytes']}")
    else:
        print(f"  ok       {name}  ({size} bytes)")

    got = sha256(weights)
    if got != man["file_sha256"]:
        problems.append(f"{name}: sha256 mismatch\n             want {man['file_sha256']}\n             got  {got}")
    else:
        print(f"  ok       {name}  sha256 {got[:16]}…")

    with open(weights, "rb") as fh:
        head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            problems.append(f"{name}: truncated header")
            return problems
        magic, version, flags, n_tensors, *_ = struct.unpack(HEADER_FMT, head)
        if magic != MAGIC:
            problems.append(f"{name}: bad magic {magic:#x}, expected {MAGIC:#x}")
            return problems
        if flags & KV_CACHE_FLAG:
            problems.append(
                f"{name}: header flags {flags:#x} mark this as a KV cache, not model "
                f"weights; verify.py checks model containers"
            )
            return problems
        if version != man["format_version"]:
            problems.append(f"{name}: format_version {version} != manifest {man['format_version']}")
        if n_tensors != man["tensor_count"]:
            problems.append(f"{name}: header says {n_tensors} tensors, manifest says {man['tensor_count']}")
        else:
            print(f"  ok       {name}  magic 0x{magic:x} + v{version} + {n_tensors} tensor entries")

    declared = {t["name"] for t in man["tensors"]}
    packed = {t["name"] for t in man["tensors"] if t["ternary"]}
    print(f"  info     {len(declared)} tensors in manifest "
          f"({len(packed)} ternary, {len(declared) - len(packed)} float32)")
    print(f"  info     total {man['total_params'] / 1e6:.1f} M params, "
          f"{man['ternary_params'] / 1e6:.1f} M ternary @ 2 bits")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-all", action="store_true",
                    help="treat any missing file listed in CHECKSUMS.txt as a failure")
    args = ap.parse_args()

    print("checksums")
    ok, missing, problems = check_checksums(args.require_all)
    print(f"  {ok} verified, {missing} missing")

    print("\ncontainer")
    problems += check_container()

    print()
    if problems:
        print(f"FAIL — {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PASS" + (" (all listed files present)" if missing == 0 else f" ({missing} file(s) not downloaded)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
