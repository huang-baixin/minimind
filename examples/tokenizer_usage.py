#!/usr/bin/env python3
"""
Tokenizer使用示例
展示如何使用不同来源的tokenizer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from dataset.lm_dataset import load_tokenizer, PretrainDataset

def demo_different_tokenizers():
    """演示使用不同tokenizer的方法"""
    
    print("=== Tokenizer使用演示 ===\n")
    
    # 方法1：使用项目默认tokenizer
    print("1. 使用项目默认tokenizer:")
    tokenizer1 = load_tokenizer()  # 不传参数，使用项目默认
    print(f"   Tokenizer类型: {type(tokenizer1).__name__}")
    print(f"   词汇表大小: {tokenizer1.vocab_size}")
    print(f"   特殊token: {tokenizer1.special_tokens_map}")
    
    # 测试分词
    text = "你好，这是一个测试句子。"
    tokens = tokenizer1.encode(text)
    print(f"   测试文本: {text}")
    print(f"   Tokenized: {tokens}")
    print(f"   解码: {tokenizer1.decode(tokens)}")
    print()
    
    # 方法2：使用Hugging Face预训练模型
    print("2. 使用Hugging Face中文BERT tokenizer:")
    tokenizer2 = load_tokenizer("bert-base-chinese")
    print(f"   Tokenizer类型: {type(tokenizer2).__name__}")
    print(f"   词汇表大小: {tokenizer2.vocab_size}")
    
    tokens = tokenizer2.encode(text)
    print(f"   测试文本: {text}")
    print(f"   Tokenized: {tokens}")
    print(f"   解码: {tokenizer2.decode(tokens)}")
    print()
    
    # 方法3：使用GPT-2 tokenizer（英文）
    print("3. 使用GPT-2 tokenizer:")
    tokenizer3 = load_tokenizer("gpt2")
    print(f"   Tokenizer类型: {type(tokenizer3).__name__}")
    print(f"   词汇表大小: {tokenizer3.vocab_size}")
    
    english_text = "Hello, this is a test sentence."
    tokens = tokenizer3.encode(english_text)
    print(f"   测试文本: {english_text}")
    print(f"   Tokenized: {tokens}")
    print(f"   解码: {tokenizer3.decode(tokens)}")
    print()

def demo_dataset_with_different_tokenizers():
    """演示使用不同tokenizer创建数据集"""
    
    print("=== 数据集与不同Tokenizer组合演示 ===\n")
    
    # 可用的tokenizer选项
    tokenizer_options = [
        (None, "项目默认tokenizer"),
        ("bert-base-chinese", "BERT中文tokenizer"),
        ("gpt2", "GPT-2英文tokenizer"),
        ("microsoft/DialoGPT-medium", "DialoGPT对话tokenizer")
    ]
    
    for tokenizer_path, description in tokenizer_options:
        print(f"使用{description}:")
        
        try:
            # 加载tokenizer
            tokenizer = load_tokenizer(tokenizer_path)
            
            # 创建数据集（使用小规模测试数据）
            dataset = PretrainDataset(
                data_path="dataset/mini_pretrain_data.jsonl",
                tokenizer=tokenizer,
                max_length=128
            )
            
            print(f"   数据集大小: {len(dataset)}")
            print(f"   Tokenizer词汇表大小: {tokenizer.vocab_size}")
            
            # 测试第一个样本
            if len(dataset) > 0:
                X, Y, loss_mask = dataset[0]
                print(f"   输入序列长度: {len(X)}")
                print(f"   目标序列长度: {len(Y)}")
                
                # 显示部分token
                sample_text = tokenizer.decode(X[:10])
                print(f"   样本预览: {sample_text[:50]}...")
            
        except Exception as e:
            print(f"   错误: {e}")
        
        print()

def compare_tokenizer_performance():
    """比较不同tokenizer的性能特点"""
    
    print("=== Tokenizer性能比较 ===\n")
    
    test_texts = [
        "自然语言处理是人工智能的重要分支。",
        "Hello world! This is a test for English text.",
        "深度学习模型需要大量的训练数据。",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    tokenizers_to_test = [
        ("项目默认", None),
        ("BERT中文", "bert-base-chinese"),
        ("GPT-2", "gpt2")
    ]
    
    for name, path in tokenizers_to_test:
        print(f"{name} tokenizer:")
        
        try:
            tokenizer = load_tokenizer(path)
            
            for text in test_texts:
                tokens = tokenizer.encode(text)
                compression_ratio = len(text) / len(tokens) if tokens else 0
                
                print(f"   文本: {text[:30]}...")
                print(f"   字符数: {len(text)}, Token数: {len(tokens)}")
                print(f"   压缩比: {compression_ratio:.2f}")
                print(f"   平均每个token字符数: {len(text)/len(tokens):.2f}")
                print()
                
        except Exception as e:
            print(f"   测试失败: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    # 运行演示
    demo_different_tokenizers()
    demo_dataset_with_different_tokenizers()
    compare_tokenizer_performance()
    
    print("=== 使用建议 ===")
    print("1. 中文任务: 推荐使用项目默认tokenizer或bert-base-chinese")
    print("2. 英文任务: 推荐使用gpt2或项目默认tokenizer")
    print("3. 对话任务: 推荐使用microsoft/DialoGPT-medium")
    print("4. 多语言任务: 使用项目默认tokenizer（已针对中文优化）")