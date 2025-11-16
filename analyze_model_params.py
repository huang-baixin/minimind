#!/usr/bin/env python3
"""
分析MiniMind模型参数量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model_minimind import MiniMindConfig, MiniMindModel
import torch

def calculate_model_params():
    """计算模型参数量"""
    
    print("=== MiniMind模型参数量分析 ===\n")
    
    # 默认配置
    config = MiniMindConfig()
    
    print("模型配置参数:")
    print(f"  - 隐藏层维度 (hidden_size): {config.hidden_size}")
    print(f"  - 层数 (num_hidden_layers): {config.num_hidden_layers}")
    print(f"  - 注意力头数 (num_attention_heads): {config.num_attention_heads}")
    print(f"  - KV头数 (num_key_value_heads): {config.num_key_value_heads}")
    print(f"  - 词汇表大小 (vocab_size): {config.vocab_size}")
    print(f"  - 中间层维度 (intermediate_size): {config.intermediate_size or config.hidden_size * 4}")
    
    # 创建模型
    model = MiniMindModel(config)
    
    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    # 按层分解参数
    print("\n参数分解:")
    
    # 嵌入层参数
    embedding_params = config.vocab_size * config.hidden_size
    print(f"  - 嵌入层: {embedding_params:,} ({embedding_params / total_params * 100:.1f}%)")
    
    # 注意力层参数
    head_dim = config.hidden_size // config.num_attention_heads
    attention_params_per_layer = (
        config.hidden_size * config.hidden_size * 4  # QKV投影 + 输出投影
    )
    attention_total = attention_params_per_layer * config.num_hidden_layers
    print(f"  - 注意力层: {attention_total:,} ({attention_total / total_params * 100:.1f}%)")
    
    # FFN层参数
    ffn_intermediate = config.intermediate_size or config.hidden_size * 4
    ffn_params_per_layer = (
        config.hidden_size * ffn_intermediate +  # 上投影
        ffn_intermediate * config.hidden_size    # 下投影
    )
    ffn_total = ffn_params_per_layer * config.num_hidden_layers
    print(f"  - FFN层: {ffn_total:,} ({ffn_total / total_params * 100:.1f}%)")
    
    # 归一化层参数
    norm_params_per_layer = config.hidden_size * 2  # RMSNorm权重
    norm_total = norm_params_per_layer * config.num_hidden_layers
    print(f"  - 归一化层: {norm_total:,} ({norm_total / total_params * 100:.1f}%)")
    
    # LM Head参数（与嵌入层共享）
    lm_head_params = 0  # 共享权重
    print(f"  - LM Head: {lm_head_params:,} (共享嵌入层)")
    
    return total_params, config

def compare_with_other_models():
    """与其他模型对比"""
    
    print("\n=== 与其他模型对比 ===\n")
    
    models = {
        "MiniMind (默认)": {"params": 25.8e6, "vocab": 6400, "layers": 8, "hidden": 512},
        "MiniMind2-Small": {"params": 26e6, "vocab": 6400, "layers": 8, "hidden": 512},
        "GPT-2 Small": {"params": 124e6, "vocab": 50257, "layers": 12, "hidden": 768},
        "GPT-2 Medium": {"params": 355e6, "vocab": 50257, "layers": 24, "hidden": 1024},
        "Qwen2-0.5B": {"params": 0.5e9, "vocab": 151936, "layers": 24, "hidden": 1024},
        "Qwen2-1.5B": {"params": 1.5e9, "vocab": 151936, "layers": 24, "hidden": 1536},
        "Qwen2-7B": {"params": 7e9, "vocab": 151936, "layers": 32, "hidden": 4096},
    }
    
    print("模型名称           | 参数量    | 词汇表 | 层数 | 隐藏维度")
    print("-" * 60)
    
    for name, info in models.items():
        params_str = f"{info['params']/1e6:.1f}M" if info['params'] < 1e9 else f"{info['params']/1e9:.1f}B"
        print(f"{name:<16} | {params_str:>8} | {info['vocab']:>6,} | {info['layers']:>4} | {info['hidden']:>8}")

def qwen_tokenizer_recommendation():
    """Qwen tokenizer推荐"""
    
    print("\n=== Qwen Tokenizer推荐 ===\n")
    
    print("📊 基于MiniMind模型特点，推荐使用以下Qwen tokenizer:")
    
    recommendations = [
        {
            "模型": "Qwen2-1.5B",
            "理由": "词汇表大小151,936，与MiniMind的6,400相比更丰富，能更好处理中文",
            "优势": "支持更细粒度的中文分词，词汇覆盖更全面",
            "注意": "需要调整模型配置以匹配词汇表大小"
        },
        {
            "模型": "Qwen2-0.5B", 
            "理由": "相对较小的模型，词汇表相同但模型更轻量",
            "优势": "部署成本低，适合资源受限环境",
            "注意": "词汇表较大，可能增加嵌入层参数"
        },
        {
            "模型": "Qwen/Qwen2-7B-Chat",
            "理由": "聊天优化版本，对话能力更强",
            "优势": "经过对话数据训练，对话效果更好",
            "注意": "模型较大，需要更多计算资源"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['模型']}:")
        print(f"   📝 {rec['理由']}")
        print(f"   ✅ {rec['优势']}")
        print(f"   ⚠️  {rec['注意']}")
    
    print("\n🎯 最佳推荐: Qwen2-1.5B")
    print("   - 词汇表丰富，中文支持好")
    print("   - 模型大小适中，部署成本合理")
    print("   - 性能与资源消耗平衡良好")

def implementation_guide():
    """实现指南"""
    
    print("\n=== 实现指南 ===\n")
    
    print("1. 使用Qwen tokenizer的步骤:")
    print("   - 安装依赖: pip install transformers")
    print("   - 加载tokenizer: from transformers import AutoTokenizer")
    print("   - 使用代码: tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B')")
    
    print("\n2. 需要调整的配置:")
    print("   - 修改MiniMindConfig中的vocab_size为151936")
    print("   - 重新初始化嵌入层权重")
    print("   - 可能需要调整模型架构以适应更大的词汇表")
    
    print("\n3. 注意事项:")
    print("   - 嵌入层参数会显著增加 (从3.3M增加到78M)")
    print("   - 需要更多显存和训练时间")
    print("   - 但能获得更好的中文处理能力")

if __name__ == "__main__":
    # 计算参数量
    total_params, config = calculate_model_params()
    
    # 对比其他模型
    compare_with_other_models()
    
    # Qwen tokenizer推荐
    qwen_tokenizer_recommendation()
    
    # 实现指南
    implementation_guide()
    
    print("\n=== 总结 ===")
    print(f"✅ MiniMind默认配置参数量: {total_params / 1e6:.1f}M")
    print("✅ 推荐使用Qwen2-1.5B的tokenizer")
    print("✅ 需要调整vocab_size配置以匹配Qwen tokenizer")
    print("✅ 嵌入层参数会增加，但能获得更好的中文处理能力")