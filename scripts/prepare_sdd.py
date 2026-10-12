"""Build the SDD transfer pairs A2S ... E2S used for Table 5 of the paper.

Each pair X2S trains on the source scene X of the ETH/UCY domain-adaptation pairs (the same files as
X2Y/train) and uses the SDD validation and test files in ETH/UCY format released with Social-Implicit
(https://github.com/abduallahmohamed/Social-Implicit, datasets/sdd/{val,test}).

    python scripts/prepare_sdd.py --sdd /path/to/Social-Implicit/datasets/sdd [--data datasets]
"""
import argparse
import os
import shutil

SOURCES = 'ABCDE'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sdd', required=True, help='datasets/sdd of a Social-Implicit clone')
    ap.add_argument('--data', default=os.environ.get('STCRF_DATA', 'datasets'))
    a = ap.parse_args()
    for src in SOURCES:
        tgt = next(t for t in SOURCES if t != src)
        dst = os.path.join(a.data, f'{src}2S')
        for phase, from_dir in (('train', os.path.join(a.data, f'{src}2{tgt}', 'train')),
                                ('val', os.path.join(a.sdd, 'val')), ('test', os.path.join(a.sdd, 'test'))):
            os.makedirs(os.path.join(dst, phase), exist_ok=True)
            files = sorted(f for f in os.listdir(from_dir) if f.endswith('.txt'))
            for f in files:
                shutil.copy2(os.path.join(from_dir, f), os.path.join(dst, phase, f))
            print(f'{dst}/{phase}: {len(files)} files from {from_dir}')
