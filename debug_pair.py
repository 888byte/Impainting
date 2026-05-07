from pathlib import Path
base=Path(r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹")

gt_dir=base/'gt'
method_path=base/'LRDiff'

gt_files=sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
method_files=[p for p in method_path.iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')]
print('gt count',len(gt_files))
print('method count',len(method_files))

method_map={p.stem:p for p in method_files}
paired=[]
for g in gt_files:
    key=g.stem
    found=None
    if key in method_map:
        found=method_map[key]
    else:
        for mf in method_files:
            if mf.stem.startswith(key):
                found=mf
                break
    if found is not None:
        paired.append((g,found))

print('paired',len(paired))
for i in range(min(20,len(paired))):
    print(paired[i][0].name,'<->',paired[i][1].name)
