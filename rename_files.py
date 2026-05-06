#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
from collections import defaultdict
from pathlib import Path

base_path = r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹"

def standardize_folder(folder_name):
    """
    标准化文件夹中的文件名，去掉前后缀
    标准格式: XXXXXX_direction.png
    """
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return False
    
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing: {folder_name}")
    print(f"Total image files: {len(files)}")
    
    renamed_count = 0
    directions = ['bottom', 'center', 'left', 'right', 'top']
    
    for file in sorted(files):
        # 提取6位数字编号和方向
        match = re.search(r'(\d{6})_(\w+)', file)
        if not match:
            print(f"  [SKIP] Cannot extract ID from: {file}")
            continue
        
        img_id = match.group(1)
        dir_part = match.group(2)
        
        # 从dir_part中提取方向（可能包含多个单词）
        direction = None
        for d in directions:
            if dir_part.startswith(d):
                direction = d
                break
        
        if not direction:
            print(f"  [SKIP] Cannot find direction in: {file}")
            continue
        
        # 标准格式：XXXXXX_direction.png
        new_name = f"{img_id}_{direction}.png"
        
        # 如果已经是标准格式，跳过
        if file == new_name:
            renamed_count += 1
            continue
        
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        
        # 如果新文件已存在，说明有重复
        if os.path.exists(new_path):
            print(f"  [WARN] Target already exists: {new_name}, skipping {file}")
            continue
        
        try:
            os.rename(old_path, new_path)
            renamed_count += 1
            print(f"  {file:35} -> {new_name}")
        except Exception as e:
            print(f"  [ERROR] Failed to rename {file}: {e}")
    
    print(f"Renamed: {renamed_count}/{len(files)}")
    
    # 最终验证
    final_files = os.listdir(folder_path)
    final_images = [f for f in final_files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 验证格式
    valid_count = 0
    for f in final_images:
        if re.match(r'\d{6}_\w+\.png$', f):
            valid_count += 1
    
    print(f"Final state: {len(final_images)} images, {valid_count} with correct format")
    return valid_count == len(final_images) == 250


def verify_all():
    """验证所有文件夹"""
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    folders = ["AdaIR", "BBAT", "RAD", "RFR"]
    all_ok = True
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 统计格式
        if folder == "RFR":
            # RFR保持img_XXX.png格式
            valid_count = sum(1 for f in files if re.match(r'img_\d+\.png$', f))
        else:
            valid_count = sum(1 for f in files if re.match(r'\d{6}_\w+\.png$', f))
        
        is_ok = len(files) == 250 and valid_count == 250
        all_ok = all_ok and is_ok
        
        status = "OK  " if is_ok else "FAIL"
        print(f"{status} {folder:10}: {len(files)} files, {valid_count} valid format")
        
        if not is_ok:
            # 显示几个不合格的文件
            for f in files[:3]:
                print(f"       - {f}")
    
    return all_ok


# 主程序
if __name__ == "__main__":
    print("Starting file renaming...")
    
    # 标准化AdaIR、BBAT、RAD
    for folder in ["AdaIR", "BBAT", "RAD"]:
        standardize_folder(folder)
    
    # RFR不需要重命名，已经是img_XXX.png格式
    print(f"\nProcessing: RFR")
    rfr_path = os.path.join(base_path, "RFR")
    rfr_files = [f for f in os.listdir(rfr_path) if f.endswith('.png')]
    print(f"Total files in RFR: {len(rfr_files)}")
    print(f"RFR format: img_XXXX.png (no changes needed)")
    
    # 验证
    verify_all()
    
    print("\nDone!")
