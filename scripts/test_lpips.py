#!/usr/bin/env python3
import sys
try:
    import torch
    import lpips
except Exception as e:
    print('Missing package or import error:', e)
    print('Run: python -m pip install --user lpips')
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
try:
    m = lpips.LPIPS(net='alex').to(device)
    print('LPIPS model loaded')
    import torch
    a = torch.randn(1,3,64,64).to(device)
    b = torch.randn(1,3,64,64).to(device)
    with torch.no_grad():
        v = m(a,b)
    print('LPIPS forward OK, value=', float(v.cpu().numpy()))
except Exception as e:
    print('LPIPS load/forward failed:', e)
    sys.exit(1)
