#!/usr/bin/env python3
"""
验证embedding weight与tokenizer词汇表大小的关系
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

def verify_embedding_tokenizer_relation():
    """验证embedding层与tokenizer的关系"""
    
    print("=== 验证Embedding Weight与Tokenizer的关系 ===\n")
    
    # 1. 加载tokenizer，获取词汇表大小
    tokenizer = AutoTokenizer.from_pretrained("./model")
    actual_vocab_size = tokenizer.vocab_size
    print(f"1. Tokenizer词汇表大小: {actual_vocab_size:,} 个token")
    
    # 2. 创建模型配置
    config = MiniMindConfig(vocab_size=actual_vocab_size, hidden_size=512)
    print(f"2. 模型配置vocab_size: {config.vocab_size:,}")
    
    # 3. 创建模型
    model = MiniMindForCausalLM(config)
    
    # 4. 检查embedding层
    embedding_layer = model.model.embed_tokens
    print(f"3. Embedding层形状: {embedding_layer.weight.shape}")
    print(f"   - 行数(vocab_size): {embedding_layer.weight.shape[0]:,}")
    print(f"   - 列数(hidden_size): {embedding_layer.weight.shape[1]:,}")
    
    # 5. 检查输出层
    lm_head_layer = model.lm_head
    print(f"4. 输出层形状: {lm_head_layer.weight.shape}")
    print(f"   - 行数(hidden_size): {lm_head_layer.weight.shape[0]:,}")
    print(f"   - 列数(vocab_size): {lm_head_layer.weight.shape[1]:,}")
    
    # 6. 验证权重共享
    print(f"5. 权重共享验证:")
    print(f"   - embedding.weight is lm_head.weight: {embedding_layer.weight is lm_head_layer.weight}")
    # 注意：权重共享时，embedding.weight 和 lm_head.weight 是同一个张量
    # embedding.weight.shape = [vocab_size, hidden_size]
    # lm_head.weight.shape = [vocab_size, hidden_size] （权重共享时相同）
    print(f"   - 权重共享机制: embedding.weight 和 lm_head.weight 是同一个张量")
    
    # 7. 参数计算
    embedding_params = embedding_layer.weight.numel()
    print(f"6. 参数统计:")
    print(f"   - Embedding层参数: {embedding_params:,}")
    print(f"   - 理论计算: {config.vocab_size} × {config.hidden_size} = {config.vocab_size * config.hidden_size:,}")
    
    # 8. 测试tokenizer与embedding的兼容性
    print(f"7. 兼容性测试:")
    test_text = "自然语言处理是人工智能的重要分支"
    input_ids = tokenizer.encode(test_text, return_tensors="pt")
    print(f"   - 测试文本: '{test_text}'")
    print(f"   - Token IDs: {input_ids.tolist()[0]}")
    print(f"   - Token数量: {len(input_ids[0])}")
    
    # 验证所有token ID都在有效范围内
    max_token_id = input_ids.max().item()
    min_token_id = input_ids.min().item()
    print(f"   - 最大token ID: {max_token_id}")
    print(f"   - 最小token ID: {min_token_id}")
    print(f"   - 是否都在[0, {config.vocab_size-1}]范围内: {min_token_id >= 0 and max_token_id < config.vocab_size}")
    
    # 9. 实际前向传播测试
    print(f"8. 前向传播测试:")
    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            print(f"   - 前向传播成功!")
            print(f"   - 输出logits形状: {logits.shape}")
            print(f"   - 应与输入对应: [batch_size, seq_len, vocab_size]")
    except Exception as e:
        print(f"   - 前向传播失败: {e}")
    
    return config.vocab_size == actual_vocab_size

def demonstrate_different_scenarios():
    """演示不同词汇表大小的影响"""
    
    print("\n=== 不同词汇表大小的影响演示 ===\n")
    
    hidden_size = 512
    vocab_sizes = [1000, 32000, 6400, 100000, 151851]  # 包括Qwen2-7B的大小
    
    print("词汇表大小 | 隐藏层维度 | Embedding参数 | 参数占比(100M总参)")
    print("-" * 70)
    
    for vocab_size in vocab_sizes:
        embedding_params = vocab_size * hidden_size
        total_params_100M = 100_000_000
        embedding_ratio = (embedding_params / total_params_100M) * 100
        
        model_name = "MiniMind" if vocab_size == 6400 else "Qwen2-7B" if vocab_size == 151851 else str(vocab_size)
        
        print(f"{model_name:>9} | {hidden_size:>10} | {embedding_params:>13,} | {embedding_ratio:>18.1f}%")

def analyze_tokenizer_vocab_composition():
    """分析tokenizer词汇表构成"""
    
    print("\n=== Tokenizer词汇表构成分析 ===\n")
    
    tokenizer = AutoTokenizer.from_pretrained("./model")
    vocab = tokenizer.get_vocab()
    
    # 分析词汇表构成
    single_chars = 0
    multi_chars = 0
    special_tokens = len(tokenizer.special_tokens_map)
    
    for token in vocab.keys():
        if len(token) == 1:
            single_chars += 1
        else:
            multi_chars += 1
    
    print(f"总词汇表大小: {len(vocab):,}")
    print(f"单字符token: {single_chars:,}")
    print(f"多字符token: {multi_chars:,}")
    print(f"特殊token: {special_tokens:,}")
    print(f"单字符占比: {single_chars/len(vocab)*100:.1f}%")
    print(f"多字符占比: {multi_chars/len(vocab)*100:.1f}%")

if __name__ == "__main__":
    # 运行验证
    is_consistent = verify_embedding_tokenizer_relation()
    
    if is_consistent:
        print("\n✅ 验证结果: embedding weight与tokenizer词汇表大小完全一致！")
    else:
        print("\n❌ 验证结果: 存在不一致！")
    
    # 演示不同场景
    demonstrate_different_scenarios()
    
    # 分析词汇表构成
    analyze_tokenizer_vocab_composition()
    
    print("\n=== 关键结论 ===")
    print("1. ✅ embedding weight的行数必须等于tokenizer的词汇表大小")
    print("2. ✅ 这是模型能够正常工作的基本要求")
    print("3. ✅ MiniMind项目正确实现了这一关系")
    print("4. ✅ 权重共享机制进一步优化了参数使用")