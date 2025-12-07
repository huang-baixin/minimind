#!/usr/bin/env python3
"""
IDE调试辅助脚本 - 为Trae IDE的调试功能提供正确的Python环境
用法：
1. 在IDE中右击 train_pretrain_fixed.py
2. 选择 "Run with Python Interpreter"
3. 选择这个脚本作为解释器
"""

import sys
import os
import subprocess

def setup_environment():
    """设置环境变量和路径"""
    # 添加minimind_venv的site-packages到Python路径
    venv_path = '/Users/huangbaixin/project/minimind/minimind_venv'
    site_packages = os.path.join(venv_path, 'lib', 'python3.9', 'site-packages')
    
    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    
    # 验证torch可用
    try:
        import torch
        print(f"✓ 成功加载torch {torch.__version__}")
        return True
    except ImportError as e:
        print(f"✗ torch导入失败: {e}")
        return False

def run_training():
    """运行训练脚本"""
    if not setup_environment():
        return False
    
    # 训练脚本路径
    script_path = '/Users/huangbaixin/project/minimind/train_pretrain_fixed.py'
    
    if os.path.exists(script_path):
        print(f"=== 启动训练脚本: {script_path} ===")
        
        # 运行训练脚本
        try:
            # 修改当前进程的PYTHONPATH
            os.environ['PYTHONPATH'] = os.path.dirname(script_path) + ':' + os.environ.get('PYTHONPATH', '')
            
            # 使用exec执行脚本内容
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # 在当前命名空间中执行脚本
            exec(script_content)
            return True
            
        except Exception as e:
            print(f"✗ 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"✗ 训练脚本不存在: {script_path}")
        return False

if __name__ == "__main__":
    print("=== IDE调试模式启动 ===")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    success = run_training()
    if success:
        print("✓ 训练脚本执行完成")
    else:
        print("✗ 训练脚本执行失败")
        sys.exit(1)