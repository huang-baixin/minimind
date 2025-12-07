#!/usr/bin/env python3
"""
Debug启动脚本 - 确保使用正确的虚拟环境
使用方法: python debug_python.py your_script.py
"""
import os
import sys
import subprocess

def main():
    # 添加虚拟环境的site-packages到Python路径
    venv_path = "/Users/huangbaixin/project/minimind/minimind_venv"
    site_packages = os.path.join(venv_path, "lib", "python3.9", "site-packages")
    
    if os.path.exists(site_packages):
        sys.path.insert(0, site_packages)
        print(f"✓ Added to Python path: {site_packages}")
    
    # 检查torch是否可用
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is available")
        print(f"✓ Torch location: {torch.__file__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return 1
    
    # 运行用户脚本
    if len(sys.argv) > 1:
        script = sys.argv[1]
        if os.path.exists(script):
            print(f"Running script: {script}")
            subprocess.run([sys.executable] + sys.argv[1:])
        else:
            print(f"Script not found: {script}")
            return 1
    else:
        print("Usage: python debug_python.py your_script.py")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())