#!/bin/bash

# MiniMind 训练数据下载脚本 - 使用 HuggingFace 源
echo "🚀 开始下载 MiniMind 训练数据 (HuggingFace)..."
mkdir -p dataset
cd dataset

# 创建临时目录用于下载
mkdir -p temp_download

echo "📥 正在下载预训练数据..."
wget -c -O temp_download/pretrain_hq.jsonl \
  "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_hq.jsonl" \
  --progress=bar:force

if [ $? -eq 0 ]; then
    echo "✅ pretrain_hq.jsonl 下载完成"
    mv temp_download/pretrain_hq.jsonl .
    rm -rf temp_download  # 清理临时目录
    ls -lh pretrain_hq.jsonl
else
    echo "❌ pretrain_hq.jsonl 下载失败，尝试备用链接..."
    rm -rf temp_download  # 清理临时目录
fi

echo ""
echo "📥 正在下载SFT数据..."
# 创建临时目录用于下载
mkdir -p temp_download
wget -c -O temp_download/sft_mini_512.jsonl \
  "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/sft_mini_512.jsonl" \
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