#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

base_path = r"d:\code\ky\bihua\Impainting\cmp\新建文件夹\新建文件夹"

def cleanup_non_standard_files(folder_name):
    """删除不符合标准格式的文件"""
    folder_path = os.path.join(base_path, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return
    
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\nProcessing: {folder_name}")
    print(f"Total image files: {len(image_files)}")
    
    # 分离标准格式和非标准格式的文件
    standard_format = []
    non_standard_format = []
    
    for file in image_files:
        if re.match(r'^\d{6}_\w+\.png$', file):
            standard_format.append(file)
        else:
            non_standard_format.append(file)
    
    print(f"Standard format (XXXXXX_direction.png): {len(standard_format)}")
    print(f"Non-standard format: {len(non_standard_format)}")
    
    # 显示非标准格式文件的样本
    if non_standard_format:
        print(f"Examples of non-standard files:")
        for f in non_standard_format[:5]:
            print(f"  - {f}")
    
    # 删除非标准格式的文件
    if non_standard_format:
        print(f"\nDeleting {len(non_standard_format)} non-standard files...")
        deleted_count = 0
        for file in non_standard_format:
            file_path = os.path.join(folder_path, file)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Failed to delete {file}: {e}")
        
        print(f"Deleted: {deleted_count} files")
    
    # 最终检查
    final_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Final count: {len(final_files)} files")
    
    return len(final_files)


def verify_final_state():
    """验证最终状态"""
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    folders = ["AdaIR", "BBAT", "RAD", "RFR"]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if folder == "RFR":
            valid_format = sum(1 for f in files if re.match(r'img_\d+\.png$', f))
        else:
            valid_format = sum(1 for f in files if re.match(r'^\d{6}_\w+\.png$', f))
        
        status = "OK" if valid_format == len(files) else "FAIL"
        print(f"{status} {folder:10}: {len(files):3} files, {valid_format:3} valid format")


# 主程序
if __name__ == "__main__":
    print("Cleaning up non-standard format files...")
    
    for folder in ["AdaIR", "BBAT", "RAD"]:
        cleanup_non_standard_files(folder)
    
    # 验证最终状态
    verify_final_state()
    
    print("\nDone!")
