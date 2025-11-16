#!/usr/bin/env python3
"""
分析tokenizer效率：词汇表大小与模型参数的关系
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
import json

def analyze_tokenizer_params(tokenizer_path):
    """分析tokenizer的参数影响"""
    
    # 加载tokenizer
    if tokenizer_path == "qwen":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        tokenizer_name = "Qwen2-7B"
    elif tokenizer_path == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer_name = "GPT-2"
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer_name = "MiniMind"
    
    vocab_size = tokenizer.vocab_size
    
    print(f"=== {tokenizer_name} Tokenizer分析 ===")
    print(f"词汇表大小: {vocab_size:,} 个token")
    print(f"特殊token数量: {len(tokenizer.special_tokens_map)}")
    
    # 不同隐藏层维度下的参数计算
    hidden_sizes = [256, 512, 768, 1024, 2048, 4096]
    
    print("\n嵌入层参数计算:")
    print("隐藏层维度 | 嵌入层参数 | 参数占比(假设总参100M)")
    print("-" * 60)
    
    for hidden_size in hidden_sizes:
        embedding_params = vocab_size * hidden_size
        total_params_100M = 100_000_000
        embedding_ratio = (embedding_params / total_params_100M) * 100
        
        print(f"{hidden_size:>10} | {embedding_params:>11,} | {embedding_ratio:>6.1f}%")
    
    return vocab_size, tokenizer

def test_compression_efficiency(tokenizer, sample_texts):
    """测试tokenizer的压缩效率"""
    
    print(f"\n=== Tokenizer压缩效率测试 ===")
    
    results = []
    for text in sample_texts:
        char_count = len(text)
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        
        if token_count > 0:
            compression_ratio = char_count / token_count
            chars_per_token = char_count / token_count
        else:
            compression_ratio = 0
            chars_per_token = 0
        
        results.append({
            'text': text[:50] + "..." if len(text) > 50 else text,
            'chars': char_count,
            'tokens': token_count,
            'compression_ratio': compression_ratio,
            'chars_per_token': chars_per_token
        })
    
    # 打印结果
    print("文本示例 | 字符数 | Token数 | 压缩比 | 每Token字符数")
    print("-" * 70)
    
    for result in results:
        print(f"{result['text']:30} | {result['chars']:>6} | {result['tokens']:>7} | {result['compression_ratio']:>7.2f} | {result['chars_per_token']:>13.2f}")
    
    # 计算平均值
    avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
    avg_chars_per_token = sum(r['chars_per_token'] for r in results) / len(results)
    
    print(f"\n平均压缩比: {avg_ratio:.2f}")
    print(f"平均每Token字符数: {avg_chars_per_token:.2f}")
    
    return avg_ratio, avg_chars_per_token

def compare_different_tokenizers():
    """比较不同tokenizer的效率"""
    
    # 测试文本
    sample_texts = [
        "自然语言处理是人工智能的重要分支。",
        "深度学习模型需要大量的训练数据才能达到好的效果。",
        "Hello world! This is a test for English text processing.",
        "机器学习算法包括监督学习、无监督学习和强化学习。",
        "The quick brown fox jumps over the lazy dog.",
        "中文分词是自然语言处理中的基础任务之一。"
    ]
    
    tokenizers_to_test = [
        ("MiniMind", "./model"),
        ("Qwen2-7B", "qwen"),
        ("GPT-2", "gpt2")
    ]
    
    results = {}
    
    for name, path in tokenizers_to_test:
        print("\n" + "="*80)
        vocab_size, tokenizer = analyze_tokenizer_params(path)
        avg_ratio, avg_chars = test_compression_efficiency(tokenizer, sample_texts)
        
        results[name] = {
            'vocab_size': vocab_size,
            'avg_compression_ratio': avg_ratio,
            'avg_chars_per_token': avg_chars
        }
    
    # 总结比较
    print("\n" + "="*80)
    print("=== Tokenizer效率比较总结 ===")
    print("Tokenizer | 词汇表大小 | 平均压缩比 | 平均每Token字符数 | 参数效率")
    print("-" * 85)
    
    for name, result in results.items():
        # 参数效率 = 压缩比 / (词汇表大小 / 1000)
        param_efficiency = result['avg_compression_ratio'] / (result['vocab_size'] / 1000)
        
        print(f"{name:9} | {result['vocab_size']:>10,} | {result['avg_compression_ratio']:>11.2f} | {result['avg_chars_per_token']:>16.2f} | {param_efficiency:>10.3f}")

def analyze_minimind_tokenizer_details():
    """详细分析MiniMind tokenizer的词汇表构成"""
    
    print("\n" + "="*80)
    print("=== MiniMind Tokenizer详细分析 ===")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model")
    
    # 分析词汇表构成
    vocab = tokenizer.get_vocab()
    
    # 统计不同类型token的数量
    chinese_chars = 0
    english_words = 0
    symbols = 0
    numbers = 0
    
    for token in vocab.keys():
        if len(token) == 1:  # 单字符
            if '\u4e00' <= token <= '\u9fff':  # 中文字符
                chinese_chars += 1
            elif token.isalpha() and token.isascii():  # 英文字母
                english_words += 1
            elif token.isdigit():  # 数字
                numbers += 1
            else:  # 符号
                symbols += 1
        else:  # 多字符（可能是词语）
            # 简单判断：如果包含中文字符，认为是中文词语
            if any('\u4e00' <= char <= '\u9fff' for char in token):
                chinese_chars += 1
            else:
                english_words += 1
    
    print(f"总词汇表大小: {len(vocab):,}")
    print(f"中文字符/词语: {chinese_chars:,}")
    print(f"英文单词: {english_words:,}")
    print(f"数字: {numbers:,}")
    print(f"符号: {symbols:,}")
    
    # 计算中文占比
    chinese_ratio = chinese_chars / len(vocab) * 100
    print(f"中文内容占比: {chinese_ratio:.1f}%")

if __name__ == "__main__":
    # 运行分析
    compare_different_tokenizers()
    analyze_minimind_tokenizer_details()
    
    print("\n" + "="*80)
    print("=== 关键结论 ===")
    print("1. 词汇表大小直接影响嵌入层参数数量")
    print("2. 更大的词汇表通常有更好的压缩效率")
    print("3. 但过大的词汇表会导致参数爆炸和训练困难")
    print("4. MiniMind的6400词汇表是中文任务的合理选择")
    print("5. 参数效率 = 压缩比 / (词汇表大小/1000)，值越高越好")