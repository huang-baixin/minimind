#!/bin/bash

# MiniMind 训练数据下载脚本
# 下载推荐快速复现配置：pretrain_hq.jsonl + sft_mini_512.jsonl
# 预计成本：3块钱，训练时间：2小时

echo "🚀 开始下载 MiniMind 训练数据..."
echo "📦 下载配置：pretrain_hq.jsonl (1.6GB) + sft_mini_512.jsonl (1.2GB)"
echo "💰 预计总大小：~2.8GB"
echo ""

# 创建dataset目录
mkdir -p dataset

# 切换到dataset目录
cd dataset

# 检查是否安装了modelscope
if ! command -v modelscope &> /dev/null
then
    echo "⚠️  ModelScope CLI 未安装，正在尝试安装..."
    pip install modelscope
fi

echo "📥 正在从 ModelScope 下载预训练数据..."
# 创建临时目录用于下载
mkdir -p temp_download
# 使用 ModelScope 下载 pretrain_hq.jsonl (预训练数据)
modelscope download --dataset gongjy/minimind_dataset --local_dir ./temp_download

# 检查下载是否成功，如果成功则移动文件到当前目录
if [ $? -eq 0 ] && [ -f "temp_download/pretrain_hq.jsonl" ]; then
    echo "✅ pretrain_hq.jsonl 下载完成"
    mv temp_download/pretrain_hq.jsonl .
    rm -rf temp_download  # 清理临时目录
    ls -lh pretrain_hq.jsonl
else
    echo "❌ 通过 ModelScope CLI 下载失败"
    rm -rf temp_download  # 清理临时目录
    echo "🔄 尝试使用 wget 从 ModelScope 镜像下载..."
    
    # 备用方案：直接使用wget从ModelScope下载
    wget -c -O pretrain_hq.jsonl \
      "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/repo?Revision=master&FilePath=pretrain_hq.jsonl" \
      --progress=bar:force
      
    if [ $? -eq 0 ]; then
        echo "✅ pretrain_hq.jsonl 下载完成"
        ls -lh pretrain_hq.jsonl
    else
        echo "❌ pretrain_hq.jsonl 下载失败"
        exit 1
    fi
fi

echo ""
echo "📥 正在下载SFT数据..."
# 创建临时目录用于下载
mkdir -p temp_download
# 下载 sft_mini_512.jsonl (SFT数据)
wget -c -O temp_download/sft_mini_512.jsonl \
  "https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset/resolve/main/sft_mini_512.jsonl" \
  --progress=bar:force

if [ $? -eq 0 ]; then
    echo "✅ sft_mini_512.jsonl 下载完成"
    mv temp_download/sft_mini_512.jsonl .
    rm -rf temp_download  # 清理临时目录
    ls -lh sft_mini_512.jsonl
else
    echo "❌ sft_mini_512.jsonl 下载失败"
    rm -rf temp_download  # 清理临时目录
fi

echo ""
echo "📊 下载总结："
ls -lh

echo ""
echo "🎉 数据下载完成！现在可以开始训练了："
echo "   1. 预训练：python train_pretrain.py"
echo "   2. 监督微调：python train_full_sft.py"
echo ""
echo "💡 提示：确保已激活虚拟环境：source minimid_venv/bin/activate"