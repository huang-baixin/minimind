#!/usr/bin/env python3
"""
MiniMind预训练脚本 - 内存优化版本
使用更小的batch size和序列长度来适应GPU内存
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_tokenizer():
    """加载MiniMind原生tokenizer"""
    tokenizer_path = Path("model")
    
    # 检查tokenizer文件是否存在
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer目录不存在: {tokenizer_path}")
    
    vocab_file = tokenizer_path / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"词汇表文件不存在: {vocab_file}")
    
    # 加载词汇表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    print(f"加载词汇表成功，词汇表大小: {len(vocab)}")
    
    # 创建兼容HuggingFace接口的tokenizer类
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.vocab = vocab
            self.vocab_size = len(vocab)
            self.id_to_token = {v: k for k, v in vocab.items()}
            self.pad_token_id = 0  # 设置pad token id
            
        def encode(self, text, **kwargs):
            # 简单的字符级tokenization
            tokens = []
            for char in text:
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    # 未知字符使用空格替代
                    tokens.append(self.vocab.get(' ', self.vocab.get('Ġ', 0)))
            return tokens
            
        def decode(self, token_ids, **kwargs):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            
            text = ''
            for token_id in token_ids:
                if token_id in self.id_to_token:
                    text += self.id_to_token[token_id]
                else:
                    text += '?'
            return text
        
        def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None, **kwargs):
            """使tokenizer可调用，兼容HuggingFace接口"""
            tokens = self.encode(text)
            
            # 处理截断
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # 处理padding
            if padding == 'max_length' and max_length:
                if len(tokens) < max_length:
                    tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
            
            # 返回格式处理
            result = {'input_ids': tokens}
            
            if return_tensors == 'pt':
                result['input_ids'] = torch.tensor([tokens], dtype=torch.long)
            
            # 创建一个简单的对象来包装结果
            class EncodingResult:
                def __init__(self, data):
                    self.input_ids = data['input_ids']
                    
                def squeeze(self):
                    if isinstance(self.input_ids, torch.Tensor):
                        return self.input_ids.squeeze()
                    return self.input_ids
            
            return EncodingResult(result)
    
    return SimpleTokenizer(vocab)


def main():
    # 内存优化参数
    epochs = 1
    batch_size = 2  # 更小的batch size
    learning_rate = 5e-4
    max_length = 256  # 更短的序列长度
    save_dir = "test_out"
    data_path = "dataset/mini_pretrain_data.jsonl"
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置分布式训练
    rank, world_size, local_rank = setup_distributed()
    
    # 加载tokenizer
    print("正在加载tokenizer...")
    tokenizer = load_tokenizer()
    
    # 模型配置
    config = MiniMindConfig(
        vocab_size=tokenizer.vocab_size,  # 使用tokenizer的实际词汇表大小
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=2048,
    )
    
    # 创建模型
    model = MiniMindForCausalLM(config)
    
    # 移动到GPU
    if torch.cuda.is_available():
        model = model.cuda(local_rank)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 数据加载
    print("正在加载数据集...")
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 训练循环
    model.train()
    total_loss = 0
    
    print(f"开始训练，共{len(dataloader)}个batch...")
    print(f"配置: batch_size={batch_size}, max_length={max_length}")
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # 处理数据格式 (X, Y, loss_mask)
            X, Y, loss_mask = batch
            
            # 移动到GPU
            if torch.cuda.is_available():
                X = X.cuda(local_rank)
                Y = Y.cuda(local_rank)
                loss_mask = loss_mask.cuda(local_rank)
            
            # 前向传播
            with autocast():
                outputs = model(
                    input_ids=X,
                    labels=Y
                )
                
                # 计算损失 (交叉熵损失)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                    ignore_index=0  # 忽略pad token
                )
            
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # 打印进度
            if batch_idx % 5 == 0:  # 更频繁地打印进度
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存模型
    if rank == 0:
        model_save_path = os.path.join(save_dir, "pretrain_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存到: {model_save_path}")
    
    print("预训练完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()