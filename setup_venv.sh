#!/bin/bash

# MiniMind 虚拟环境手动安装脚本
# 解决 CREATE_VENV.PIP_FAILED_INSTALL_REQUIREMENTS 错误

echo "正在创建虚拟环境..."
python3 -m venv minimind_venv

# 激活虚拟环境
if [ -f "minimind_venv/bin/activate" ]; then
    source minimind_venv/bin/activate
elif [ -f "minimind_venv/Scripts/activate" ]; then
    source minimind_venv/Scripts/activate
else
    echo "错误：无法找到虚拟环境激活脚本"
    exit 1
fi

echo "升级pip..."
pip install --upgrade pip

echo "安装核心依赖（分步安装避免冲突）..."

# 第一步：安装PyTorch（使用清华镜像源加速）
echo "安装PyTorch..."
pip install torch==2.0.1 torchvision==0.15.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 第二步：安装transformers和相关NLP包
echo "安装NLP相关包..."
pip install transformers==4.30.2 datasets==2.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 第三步：安装其他核心依赖
echo "安装其他核心依赖..."
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 scikit-learn==1.3.0

# 第四步：安装项目特定依赖
echo "安装项目特定依赖..."
pip install peft==0.3.0 trl==0.4.7 jsonlines==3.1.0 jieba==0.42.1

# 第五步：安装可选依赖
echo "安装可选依赖..."
pip install flask==2.3.3 flask-cors==4.0.0 streamlit==1.27.0 wandb==0.15.8

# 第六步：安装剩余依赖
echo "安装剩余依赖..."
pip install rich==13.5.2 psutil==5.9.5 nltk==3.8.1 ujson==5.8.0

echo "安装完成！"
echo ""
echo "激活虚拟环境命令："
echo "source minimind_venv/bin/activate"
echo ""
echo "验证安装："
echo "python3 -c 'import torch; print(f\"PyTorch版本: {torch.__version__}\")'"
echo "python3 -c 'import transformers; print(f\"Transformers版本: {transformers.__version__}\")'"