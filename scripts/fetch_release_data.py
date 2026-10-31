#!/usr/bin/env python3
"""Fetch and verify the complete v1.0.0 release into a new directory.

The smaller browsable Git tree deliberately omits large raw records. This
helper restores the complete immutable release, without overwriting a checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

URL='https://github.com/pengliangtian06-commits/decoded-risk-mimo-ofdm/releases/download/v1.0.0/decoded-risk-mimo-ofdm-v1.0.0.zip'
SHA256='479407ccdf67bafb72cbc07a47c313e12f62a0da6e5dace5e064461fec3f9277'

def extract_checked(archive,output):
    if output.exists():
        raise FileExistsError('Choose a new --output directory; existing paths are never overwritten')
    actual=hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual!=SHA256:
        raise ValueError('Release archive SHA-256 mismatch')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('ARCHIVE_MANIFEST.json'))
        expected=set(manifest['files'])|{'ARCHIVE_MANIFEST.json'}
        if len(z.namelist())!=len(expected) or set(z.namelist())!=expected:
            raise ValueError('Unexpected or duplicated archive members')
        for name,digest in manifest['files'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe member path')
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:
                raise ValueError('Payload checksum mismatch: '+name)
        output.mkdir(parents=True,exist_ok=False)
        z.extractall(output)
    return len(manifest['files'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('release-v1.0.0'))
    p.add_argument('--archive',type=Path,help='Verify an already downloaded official archive instead of fetching')
    a=p.parse_args();output=a.output.resolve()
    if output.exists():raise FileExistsError(output)
    if a.archive:
        count=extract_checked(a.archive.resolve(),output)
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mimo-release-',dir=output.parent) as temp:
            archive=Path(temp)/'release.zip'
            request=urllib.request.Request(URL,headers={'User-Agent':'decoded-risk-mimo-ofdm release verifier'})
            with urllib.request.urlopen(request,timeout=60) as response,archive.open('wb') as handle:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    handle.write(chunk)
                    if handle.tell()>100_000_000:raise ValueError('Unexpected release size')
            count=extract_checked(archive,output)
    print('Verified archive SHA-256 and',count,'payload hashes.')
    print('Complete release:',output)
    print('Next: activate the pinned Python environment, then run')
    print('python',output/'scripts/reproduce_archive.py')

if __name__=='__main__':main()
