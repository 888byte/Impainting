#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

base_path = r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹"
folders = ["AdaIR", "BBAT", "LRDiff", "PGRDiff", "PromptIR", "RAD", "RePaint", "RFR"]

# 首先分析cada folder的文件结构
def analyze_folder(folder_name):
    folder_path = os.path.join(base_path, folder_name)
    files = os.listdir(folder_path)
    
    print(f"\n=== {folder_name} ===")
    print(f"Total files: {len(files)}")
    
    # 按照编号分组
    groups = defaultdict(list)
    
    for file in files:
        # 尝试提取编号
        match = re.match(r'(\d{6}|img_(\d+))', file)
        if match:
            if match.group(1).startswith('img_'):
                img_num = match.group(2)
                groups[img_num].append(file)
            else:
                num = match.group(1)
                groups[num].append(file)
        else:
            print(f"  Unmatched: {file}")
    
    # 显示统计信息
    unique_nums = len(groups)
    print(f"Unique image groups: {unique_nums}")
    
    # 显示每组的文件数
    file_counts = defaultdict(int)
    for num, files_in_group in groups.items():
        file_counts[len(files_in_group)] += 1
    
    print(f"File counts distribution: {dict(file_counts)}")
    
    # 显示一些样例
    sample_nums = sorted(list(groups.keys()))[:3]
    for num in sample_nums:
        print(f"  {num}: {groups[num]}")
    
    return groups

# 分析所有文件夹
for folder in folders:
    analyze_folder(folder)
