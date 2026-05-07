#!/usr/bin/env python3
import os
import ssl
import sys
import urllib.request
from pathlib import Path

home = Path.home()
ckpt_dir = home / '.cache' / 'torch' / 'hub' / 'checkpoints'
ckpt_dir.mkdir(parents=True, exist_ok=True)

urls = {
    'alexnet-owt-7be5be79.pth': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
    'vgg16-397923af.pth': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

for name, url in urls.items():
    dst = ckpt_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f'Skipping (exists): {dst}')
        continue
    print(f'Downloading {name} -> {dst}')
    try:
        urllib.request.urlretrieve(url, dst)
        print('  OK')
    except Exception as e:
        print('  Direct download failed:', e)
        print('  Retrying with unverified SSL...')
        try:
            ctx = ssl._create_unverified_context()
            urllib.request.urlretrieve(url, dst, context=ctx)
            print('  OK (unverified SSL)')
        except Exception as e2:
            print('  Failed to download', name)
            print('  Please download manually from:', url)
            print('  And place it at:', dst)
            sys.exit(1)

print('\nAll done. Check the files in:', ckpt_dir)
