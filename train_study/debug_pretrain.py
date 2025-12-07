#!/usr/bin/env python3
"""
预训练流程调试脚本
"""

import os
import sys
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

def debug_pretrain():
    print("=== MiniMind 预训练调试 ===")
    
    # 1. 检查CUDA可用性
    print("1. 检查CUDA设备...")
    if torch.cuda.is_available():
        print(f"   GPU可用: {torch.cuda.get_device_name()}")
        print(f"   CUDA版本: {torch.version.cuda}")
        device = torch.device("cuda:0")
    else:
        print("   GPU不可用，使用CPU")
        device = torch.device("cpu")
    
    # 2. 加载tokenizer
    print("\n2. 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./model")
        print(f"   Tokenizer加载成功")
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        print(f"   特殊token: {tokenizer.special_tokens_map}")
    except Exception as e:
        print(f"   Tokenizer加载失败: {e}")
        return
    
    # 3. 创建模型配置
    print("\n3. 创建模型配置...")
    config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        vocab_size=tokenizer.vocab_size
    )
    print(f"   隐藏层维度: {config.hidden_size}")
    print(f"   隐藏层数量: {config.num_hidden_layers}")
    print(f"   词汇表大小: {config.vocab_size}")
    
    # 4. 创建模型
    print("\n4. 创建模型...")
    try:
        model = MiniMindForCausalLM(config)
        model = model.to(device)
        print(f"   模型创建成功")
        print(f"   总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    except Exception as e:
        print(f"   模型创建失败: {e}")
        return
    
    # 5. 加载数据集
    print("\n5. 加载数据集...")
    try:
        dataset = PretrainDataset("dataset/mini_pretrain_data.jsonl", tokenizer, max_length=128)
        print(f"   数据集加载成功")
        print(f"   样本数量: {len(dataset)}")
        
        # 测试一个样本
        X, Y, loss_mask = dataset[0]
        print(f"   输入形状: {X.shape}")
        print(f"   目标形状: {Y.shape}")
        print(f"   掩码形状: {loss_mask.shape}")
        
        # 解码样本查看内容
        input_text = tokenizer.decode(X.tolist(), skip_special_tokens=False)
        print(f"   样本内容: {input_text[:100]}...")
        
    except Exception as e:
        print(f"   数据集加载失败: {e}")
        return
    
    # 6. 前向传播测试
    print("\n6. 前向传播测试...")
    try:
        model.eval()
        with torch.no_grad():
            # 准备输入数据
            X_batch = X.unsqueeze(0).to(device)
            Y_batch = Y.unsqueeze(0).to(device)
            
            # 前向传播
            outputs = model(X_batch)
            
            print(f"   前向传播成功")
            print(f"   输出logits形状: {outputs.logits.shape}")
            print(f"   损失值: {outputs.loss if hasattr(outputs, 'loss') else 'N/A'}")
            print(f"   辅助损失: {outputs.aux_loss if hasattr(outputs, 'aux_loss') else 'N/A'}")
            
    except Exception as e:
        print(f"   前向传播失败: {e}")
        print("   尝试使用CPU进行调试...")
        
        try:
            # 在CPU上重试
            model_cpu = MiniMindForCausalLM(config)
            X_cpu = X.unsqueeze(0)
            outputs_cpu = model_cpu(X_cpu)
            print(f"   CPU前向传播成功")
            print(f"   CPU输出logits形状: {outputs_cpu.logits.shape}")
        except Exception as e2:
            print(f"   CPU前向传播也失败: {e2}")
    
    print("\n=== 调试完成 ===")
    print("如果所有步骤都成功，可以开始正式预训练")

if __name__ == "__main__":
    debug_pretrain()