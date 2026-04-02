#!/usr/bin/env python3
"""
环境检查脚本
检查 Python 版本、PyTorch 安装情况、CUDA 可用性及 GPU 信息
"""

import sys
import platform


def main():
    print("=" * 50)
    print("环境检查报告")
    print("=" * 50)

    # 1. 检查 Python 版本
    print(f"\n[1] Python 版本:")
    print(f"    版本号: {sys.version}")
    print(f"    实现: {platform.python_implementation()}")
    print(f"    编译日期: {platform.python_compiler()}")

    # 2. 检查 PyTorch
    print(f"\n[2] PyTorch 检查:")
    try:
        import torch
        print(f"    PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("    [错误] PyTorch 未安装")
        return

    # 3. 检查 CUDA
    print(f"\n[3] CUDA 检查:")
    cuda_available = torch.cuda.is_available()
    print(f"    CUDA 可用: {cuda_available}")

    if cuda_available:
        print(f"    CUDA 版本: {torch.version.cuda}")
        print(f"    cuDNN 版本: {torch.backends.cudnn.version()}")

        # GPU 设备信息
        if torch.cuda.device_count() > 0:
            print(f"    GPU 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.get_device_name(i)
                # 获取显存容量（单位：GB）
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    GPU {i}: {device}")
                print(f"        显存容量: {total_memory:.2f} GB")
    else:
        print("    [提示] CUDA 不可用，将使用 CPU 运行")

    print("\n" + "=" * 50)
    print("环境检查完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
