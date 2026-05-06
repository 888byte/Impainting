#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
from collections import defaultdict
from pathlib import Path

base_path = r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹"
folders = ["AdaIR", "BBAT", "LRDiff", "PGRDiff", "PromptIR", "RAD", "RePaint", "RFR"]

# 获取标准的6位数字编号列表（从RePaint文件夹获取）
def get_standard_image_numbers():
    repaint_path = os.path.join(base_path, "RePaint")
    numbers = set()
    for file in os.listdir(repaint_path):
        match = re.match(r'(\d{6})_', file)
        if match:
            numbers.add(match.group(1))
    return sorted(list(numbers))

standard_numbers = get_standard_image_numbers()
print(f"Standard image numbers from RePaint: {standard_numbers}")
print(f"Count: {len(standard_numbers)}")

# 分析每个文件夹
def analyze_and_organize(folder_name):
    folder_path = os.path.join(base_path, folder_name)
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    
    files = os.listdir(folder_path)
    print(f"Total files: {len(files)}")
    
    # 按照image_id和方向分组
    image_groups = defaultdict(dict)
    directions = ['bottom', 'center', 'left', 'right', 'top']
    
    for file in files:
        if not file.endswith(('.png', '.jpg', '.jpeg')):
            print(f"  [SKIP] Non-image file: {file}")
            continue
        
        # 提取6位数字
        match = re.search(r'(\d{6})', file)
        if match:
            img_id = match.group(1)
            
            # 提取方向
            direction = None
            for dir_name in directions:
                if dir_name in file:
                    direction = dir_name
                    break
            
            if direction:
                image_groups[img_id][direction] = file
            else:
                print(f"  [WARN] Cannot find direction in: {file}")
        else:
            print(f"  [SKIP] Cannot find image ID in: {file}")
    
    # 统计信息
    complete_groups = sum(1 for groups in image_groups.values() if len(groups) == 5)
    incomplete_groups = len(image_groups) - complete_groups
    
    print(f"Image groups found: {len(image_groups)}")
    print(f"  Complete (5 directions): {complete_groups}")
    print(f"  Incomplete: {incomplete_groups}")
    
    # 显示不完整的组
    if incomplete_groups > 0:
        print("  Incomplete groups:")
        for img_id, dirs in sorted(image_groups.items()):
            if len(dirs) < 5:
                missing = set(directions) - set(dirs.keys())
                print(f"    {img_id}: {list(dirs.keys())} (missing: {list(missing)})")
    
    return image_groups, len(files)

# 分析所有文件夹
for folder in folders:
    groups, total_files = analyze_and_organize(folder)
