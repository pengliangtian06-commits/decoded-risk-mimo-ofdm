#!/usr/bin/env python3
"""Fetch pinned Eigen headers from the official upstream archive."""
import hashlib
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
SHA256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"


def main():
    archive = ROOT / ".cache/eigen-3.4.0.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "MIMO-OFDM research dependency retrieval"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Eigen archive checksum mismatch")
    target = ROOT / "vendor"
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(target, filter="data")
    print("Verified Eigen 3.4.0 headers:", target / "eigen-3.4.0")


if __name__ == "__main__":
    main()
