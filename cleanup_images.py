#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import re
from collections import defaultdict
from pathlib import Path

base_path = r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹"
folders = ["AdaIR", "BBAT", "RAD", "RFR"]
incomplete_folders = ["LRDiff", "PGRDiff", "PromptIR"]

def clean_and_reorganize(folder_name):
    """清理和重新组织每个文件夹中的图片"""
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return False
    
    files = os.listdir(folder_path)
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    print(f"Total files before: {len(files)}")
    
    # 按照image_id和方向分组
    image_groups = defaultdict(dict)
    directions = ['bottom', 'center', 'left', 'right', 'top']
    
    for file in files:
        if not file.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # 移除前后缀，提取编号和方向
        # 处理 AdaIR: XXX_direction_result.png
        # 处理 BBAT: Out_XXX_direction.jpg
        # 处理 RAD: 其他格式
        # 处理 RFR: img_N.png
        
        img_id = None
        direction = None
        
        # 尝试提取6位数字编号和方向
        match = re.search(r'(\d{6})_(\w+)(?:_result)?\.(?:png|jpg|jpeg)$', file)
        if match:
            img_id = match.group(1)
            potential_dir = match.group(2)
            if potential_dir in directions:
                direction = potential_dir
        
        if img_id and direction:
            image_groups[img_id][direction] = file
        else:
            print(f"  [WARN] Cannot parse: {file}")
    
    # 统计完整的分组
    complete_groups = {}
    incomplete_files = []
    
    for img_id, dirs in image_groups.items():
        if len(dirs) == 5:  # 5个方向都齐全
            complete_groups[img_id] = dirs
        else:
            for file in dirs.values():
                incomplete_files.append(file)
    
    print(f"Complete image groups (with 5 directions): {len(complete_groups)}")
    print(f"Expected: 250 images / 5 = 50 complete groups")
    
    if len(complete_groups) < 50:
        print(f"[WARNING] Only {len(complete_groups)} complete groups, need 50!")
        return False
    
    # 重命名文件为标准格式
    renamed_count = 0
    
    for img_id in sorted(complete_groups.keys()):
        dirs_dict = image_groups[img_id]
        
        for direction in directions:
            if direction not in dirs_dict:
                print(f"  [ERROR] Missing {direction} for {img_id}")
                continue
            
            old_file = dirs_dict[direction]
            old_path = os.path.join(folder_path, old_file)
            
            # 标准格式：XXXXXX_direction.png
            new_file = f"{img_id}_{direction}.png"
            new_path = os.path.join(folder_path, new_file)
            
            # 如果文件名已经是标准格式，跳过
            if old_file == new_file:
                renamed_count += 1
                continue
            
            # 重命名文件
            try:
                shutil.move(old_path, new_path)
                renamed_count += 1
                print(f"  Renamed: {old_file} -> {new_file}")
            except Exception as e:
                print(f"  [ERROR] Failed to rename {old_file}: {e}")
    
    # 删除不完整的文件
    for file in incomplete_files:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"  Deleted incomplete: {file}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {file}: {e}")
    
    # 最终检查
    final_files = os.listdir(folder_path)
    print(f"Total files after: {len(final_files)}")
    
    # 统计格式
    png_files = [f for f in final_files if f.endswith('.png')]
    print(f"PNG files: {len(png_files)}")
    
    # 验证文件名格式
    valid_count = 0
    for f in final_files:
        if re.match(r'\d{6}_\w+\.png$', f):
            valid_count += 1
        else:
            print(f"  [INVALID FORMAT] {f}")
    
    print(f"Files with correct format: {valid_count}/{len(final_files)}")
    
    return len(final_files) == 250 and valid_count == 250


def handle_rfr_folder():
    """特殊处理RFR文件夹（有500个img_X.png文件，需要减少到250个）"""
    folder_path = os.path.join(base_path, "RFR")
    files = sorted(os.listdir(folder_path))
    
    print(f"\n{'='*60}")
    print(f"Processing: RFR (special handling)")
    print(f"{'='*60}")
    print(f"Total files: {len(files)}")
    
    # img_XXX.png 格式的文件
    img_files = [f for f in files if f.startswith('img_') and f.endswith('.png')]
    print(f"Image files: {len(img_files)}")
    
    # 提取数字，按照数字排序
    img_numbers = {}
    for f in img_files:
        match = re.match(r'img_(\d+)\.png', f)
        if match:
            num = int(match.group(1))
            img_numbers[num] = f
    
    # 取前250个（img_1 到 img_250）
    to_keep = sorted(img_numbers.keys())[:250]
    to_delete = sorted(img_numbers.keys())[250:]
    
    print(f"Keeping: img_1 to img_{max(to_keep)}")
    print(f"Deleting: {len(to_delete)} files")
    
    # 删除超出的文件
    deleted_count = 0
    for num in to_delete:
        file_path = os.path.join(folder_path, img_numbers[num])
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to delete {img_numbers[num]}: {e}")
    
    print(f"Deleted: {deleted_count} files")
    
    # 删除非图片文件
    for f in files:
        if not f.startswith('img_') or not f.endswith('.png'):
            file_path = os.path.join(folder_path, f)
            try:
                os.remove(file_path)
                print(f"  Deleted non-image: {f}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {f}: {e}")
    
    final_files = os.listdir(folder_path)
    print(f"Total files after cleanup: {len(final_files)}")
    
    return len(final_files) == 250


def verify_all_folders():
    """验证所有文件夹的最终状态"""
    print(f"\n{'='*60}")
    print(f"FINAL VERIFICATION")
    print(f"{'='*60}")
    
    all_folders = ["AdaIR", "BBAT", "RAD", "RFR"]
    results = {}
    
    for folder in all_folders:
        folder_path = os.path.join(base_path, folder)
        files = os.listdir(folder_path)
        
        # 统计图片文件
        png_files = [f for f in files if f.endswith('.png')]
        jpg_files = [f for f in files if f.endswith('.jpg')]
        
        # 验证格式
        valid_format = 0
        for f in files:
            if re.match(r'\d{6}_\w+\.png$', f) or (folder == "RFR" and re.match(r'img_\d+\.png$', f)):
                valid_format += 1
        
        results[folder] = {
            'total': len(files),
            'png': len(png_files),
            'jpg': len(jpg_files),
            'valid_format': valid_format
        }
        
        status = "OK" if len(files) == 250 and valid_format == 250 else "FAIL"
        print(f"{status} {folder:12} - Total: {len(files):3}, PNG: {len(png_files):3}, JPG: {len(jpg_files):3}, Valid format: {valid_format:3}")
    
    return results


# 主程序
if __name__ == "__main__":
    print("Starting image cleanup and organization...")
    
    # 处理标准文件夹
    success_count = 0
    for folder in folders:
        if clean_and_reorganize(folder):
            success_count += 1
    
    # 特殊处理RFR
    if handle_rfr_folder():
        success_count += 1
    
    # 最终验证
    verify_all_folders()
    
    # 处理不完整的文件夹
    print(f"\n{'='*60}")
    print(f"Incomplete folders (to be handled separately or deleted):")
    for folder in incomplete_folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            files_count = len(os.listdir(folder_path))
            print(f"  {folder}: {files_count} files")
    
    print("\nCleanup complete!")
