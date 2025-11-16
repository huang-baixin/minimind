#!/usr/bin/env python3
"""
检查tokenizer和数据的兼容性
"""

import json
import torch
from transformers import AutoTokenizer

def check_tokenizer_data():
    print("=== 检查tokenizer和数据兼容性 ===")
    
    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   pad_token_id: {tokenizer.pad_token_id}")
    print(f"   bos_token_id: {tokenizer.bos_token_id}")
    print(f"   eos_token_id: {tokenizer.eos_token_id}")
    
    # 2. 检查词汇表范围
    print("\n2. 词汇表范围检查...")
    max_token_id = tokenizer.vocab_size - 1
    print(f"   最大有效token ID: {max_token_id}")
    
    # 3. 加载并检查数据
    print("\n3. 检查数据中的token ID...")
    data_file = "dataset/mini_pretrain_data.jsonl"
    
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"   数据文件: {data_file}")
    print(f"   样本数量: {len(lines)}")
    
    # 检查前几个样本
    for i, line in enumerate(lines[:5]):
        data = json.loads(line.strip())
        text = data['text']
        
        # 编码文本
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding.input_ids.squeeze()
        
        # 检查token ID范围
        max_id_in_sample = input_ids.max().item()
        min_id_in_sample = input_ids.min().item()
        
        print(f"\n   样本 {i+1}:")
        print(f"     文本: {text[:50]}...")
        print(f"     token数量: {len(input_ids)}")
        print(f"     最小token ID: {min_id_in_sample}")
        print(f"     最大token ID: {max_id_in_sample}")
        
        # 检查是否有超出范围的token
        if max_id_in_sample > max_token_id:
            print(f"     ⚠️  警告: 发现超出词汇表的token ID ({max_id_in_sample} > {max_token_id})")
            
            # 找出超出范围的token
            out_of_range_ids = input_ids[input_ids > max_token_id]
            print(f"     超出范围的token ID: {out_of_range_ids.tolist()}")
            
            # 尝试解码这些token
            for token_id in out_of_range_ids.unique():
                try:
                    token_str = tokenizer.decode([token_id.item()])
                    print(f"     token {token_id} 解码为: {repr(token_str)}")
                except:
                    print(f"     token {token_id} 无法解码")
        else:
            print(f"     ✅ token ID都在有效范围内")
    
    # 4. 检查tokenizer的特殊token
    print("\n4. 特殊token检查...")
    special_tokens = tokenizer.special_tokens_map
    print(f"   特殊token映射: {special_tokens}")
    
    # 5. 检查tokenizer配置
    print("\n5. Tokenizer配置检查...")
    config_file = "model/tokenizer_config.json"
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"   配置文件: {config_file}")
    print(f"   模型类型: {config.get('model_type', 'N/A')}")
    print(f"   填充方向: {config.get('padding_side', 'N/A')}")
    
    # 6. 检查词汇表文件
    print("\n6. 词汇表文件检查...")
    vocab_file = "model/tokenizer.json"
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab_size_from_file = len(vocab_data.get('model', {}).get('vocab', {}))
    print(f"   词汇表文件: {vocab_file}")
    print(f"   文件中的词汇表大小: {vocab_size_from_file}")
    
    if vocab_size_from_file != tokenizer.vocab_size:
        print(f"   ⚠️  警告: 词汇表大小不匹配 (文件: {vocab_size_from_file}, tokenizer: {tokenizer.vocab_size})")
    else:
        print(f"   ✅ 词汇表大小一致")

if __name__ == "__main__":
    check_tokenizer_data()