import numpy as np

def view_npz_file(npz_file_path):
    """
    控制台查看npz文件的详细内容
    :param npz_file_path: npz文件的路径（相对路径或绝对路径）
    """
    try:
        # 1. 加载npz文件（返回NpzFile对象，类似字典结构）
        npz_data = np.load(npz_file_path)
        
        # 2. 首先查看npz文件中包含的所有数组名称（keys()方法）
        print("=" * 50)
        print(f"npz文件 [{npz_file_path}] 包含的数组列表：")
        array_names = list(npz_data.keys())
        for idx, name in enumerate(array_names, 1):
            print(f"  {idx}. {name}")
        print("=" * 50)
        
        # 3. 逐个遍历数组，查看每个数组的详细信息
        for array_name in array_names:
            # 提取对应数组
            arr = npz_data[array_name]
            
            # 打印数组核心信息
            print(f"\n【数组名称】：{array_name}")
            print(f"  数组形状（shape）：{arr.shape}")
            print(f"  数组数据类型（dtype）：{arr.dtype}")
            print(f"  数组元素个数：{arr.size}")
            print(f"  数组维度（ndim）：{arr.ndim}")
            print(f"  数组内容：")
            # 优化大数组的控制台输出，避免刷屏
            if arr.size > 1000:
                print(f"    （数组过大，仅展示前5行×前5列数据）")
                print(arr[:5, :5] if arr.ndim >= 2 else arr[:5])
            else:
                print(arr)
        
        # 4. 关闭npz文件（可选，自动垃圾回收也会处理）
        npz_data.close()
        
    except FileNotFoundError:
        print(f"错误：找不到指定的npz文件，请检查路径是否正确 -> {npz_file_path}")
    except Exception as e:
        print(f"错误：读取npz文件时发生异常 -> {str(e)}")

# ---------------------- 调用示例 ----------------------
if __name__ == "__main__":
    # 替换为你的npz文件路径（相对路径或绝对路径均可）
    YOUR_NPZ_FILE_PATH = "pigment_lut33.npz"
    view_npz_file(YOUR_NPZ_FILE_PATH)