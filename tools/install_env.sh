#!/bin/bash
# MiniMind Virtual Environment Setup Script

echo "=== MiniMind 环境设置开始 ==="

# 创建虚拟环境
echo "1. 创建虚拟环境 'minimid_venv'..."
python3 -m venv minimid_venv

if [ $? -eq 0 ]; then
    echo "✓ 虚拟环境创建成功"
else
    echo "✗ 虚拟环境创建失败"
    exit 1
fi

# 激活虚拟环境并安装依赖
echo "2. 激活虚拟环境并安装依赖..."
source minimid_venv/bin/activate

echo "3. 升级 pip..."
pip install --upgrade pip

echo "4. 安装 requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ 依赖安装成功"
else
    echo "✗ 依赖安装失败"
    exit 1
fi

echo "=== 环境设置完成 ==="
echo "使用以下命令激活环境："
echo "source minimid_venv/bin/activate"
echo ""
echo "验证安装："
echo "python --version"
echo "pip list"