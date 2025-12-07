#!/usr/bin/env python3
"""
MiniMind 调试助手
用于在虚拟环境中调试代码
"""

import os
import sys
import torch
import transformers
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

def check_environment():
    """检查环境配置"""
    print("=== 环境检查 ===")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
    print()

def test_model_loading():
    """测试模型加载"""
    print("=== 模型加载测试 ===")
    try:
        # 测试配置加载
        config = MiniMindConfig()
        print("✓ 配置加载成功")
        print(f"  模型参数: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
        
        # 测试模型初始化
        model = MiniMindForCausalLM(config)
        print("✓ 模型初始化成功")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
    print()

def test_tokenizer():
    """测试分词器"""
    print("=== 分词器测试 ===")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("./model")
        print("✓ 分词器加载成功")
        
        # 测试分词
        text = "你好，这是一个测试句子"
        tokens = tokenizer.encode(text)
        print(f"  测试文本: {text}")
        print(f"  Token数量: {len(tokens)}")
        print(f"  Token IDs: {tokens}")
        
    except Exception as e:
        print(f"✗ 分词器加载失败: {e}")
    print()

def test_dataset():
    """测试数据集"""
    print("=== 数据集测试 ===")
    try:
        # 检查数据文件是否存在
        data_files = ["dataset/pretrain_data.jsonl", "dataset/mini_pretrain_data.jsonl"]
        for file in data_files:
            if os.path.exists(file):
                print(f"✓ 数据文件存在: {file}")
            else:
                print(f"⚠ 数据文件不存在: {file}")
        
        # 测试数据集加载（如果数据文件存在）
        if os.path.exists("dataset/mini_pretrain_data.jsonl"):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("./model")
            dataset = PretrainDataset("dataset/mini_pretrain_data.jsonl", tokenizer)
            print(f"✓ 数据集加载成功，样本数: {len(dataset)}")
            
            # 测试一个样本
            sample = dataset[0]
            print(f"  样本结构: {list(sample.keys())}")
            
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
    print()

def main():
    """主函数"""
    print("🚀 MiniMind 调试助手")
    print("=" * 50)
    
    check_environment()
    test_model_loading()
    test_tokenizer()
    test_dataset()
    
    print("✅ 调试完成！")
    print("\n💡 下一步建议:")
    print("1. 运行 'python debug_pretrain.py' 进行预训练调试")
    print("2. 运行 'python trainer/train_pretrain.py' 开始正式训练")
    print("3. 使用 'python -m pdb script.py' 进行调试")

if __name__ == "__main__":
    main()