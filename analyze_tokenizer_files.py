#!/usr/bin/env python3
"""
分析不同tokenizer的文件结构和使用方法
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
import json

def analyze_minimind_tokenizer():
    """分析MiniMind tokenizer的文件结构"""
    
    print("=== MiniMind Tokenizer文件分析 ===\n")
    
    # 加载MiniMind tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("./model")
        print("✅ MiniMind tokenizer加载成功")
        
        # 分析文件结构
        model_dir = "./model"
        files = os.listdir(model_dir)
        tokenizer_files = [f for f in files if 'tokenizer' in f]
        
        print("文件结构:")
        for file in tokenizer_files:
            file_path = os.path.join(model_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"  - {file} ({file_size:,} bytes)")
            
            # 简要查看文件内容
            if file == "tokenizer.json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"    - 包含: vocab字典 + BPE模型 + 预处理配置")
                    print(f"    - vocab大小: {len(data.get('model', {}).get('vocab', {}))}")
            elif file == "tokenizer_config.json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"    - 包含: 特殊token配置 + 模型参数")
                    print(f"    - 特殊token: {list(data.get('added_tokens_decoder', {}).keys())}")
        
        return True
    except Exception as e:
        print(f"❌ MiniMind tokenizer加载失败: {e}")
        return False

def analyze_qwen_tokenizer():
    """分析Qwen tokenizer的文件结构"""
    
    print("\n=== Qwen Tokenizer文件分析 ===\n")
    
    try:
        # 尝试从Hugging Face加载Qwen tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
        print("✅ Qwen tokenizer加载成功")
        
        # 获取tokenizer的配置信息
        print("Qwen tokenizer特点:")
        print(f"  - 词汇表大小: {tokenizer.vocab_size:,}")
        print(f"  - 特殊token数量: {len(tokenizer.special_tokens_map)}")
        print(f"  - 模型最大长度: {tokenizer.model_max_length}")
        
        # 如果是本地文件，分析文件结构
        print("\nQwen tokenizer通常包含的文件:")
        print("  - tokenizer.json (一体化格式)")
        print("  - tokenizer_config.json")
        print("  - special_tokens_map.json")
        
        return True
    except Exception as e:
        print(f"❌ Qwen tokenizer加载失败: {e}")
        return False

def analyze_gpt2_tokenizer():
    """分析GPT-2 tokenizer的文件结构（传统格式）"""
    
    print("\n=== GPT-2 Tokenizer文件分析（传统格式） ===\n")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✅ GPT-2 tokenizer加载成功")
        
        print("GPT-2 tokenizer文件结构（传统格式）:")
        print("  - vocab.json (词汇表映射)")
        print("  - merges.txt (BPE合并规则)")
        print("  - tokenizer_config.json")
        print("  - special_tokens_map.json")
        
        print(f"\nGPT-2 tokenizer特点:")
        print(f"  - 词汇表大小: {tokenizer.vocab_size:,}")
        print(f"  - 使用BPE算法")
        print(f"  - 需要多个文件配合使用")
        
        return True
    except Exception as e:
        print(f"❌ GPT-2 tokenizer加载失败: {e}")
        return False

def demonstrate_tokenizer_usage():
    """演示如何使用不同tokenizer"""
    
    print("\n=== Tokenizer使用演示 ===\n")
    
    test_text = "自然语言处理是人工智能的重要分支"
    
    # 1. 使用MiniMind tokenizer
    print("1. MiniMind tokenizer:")
    try:
        minimind_tokenizer = AutoTokenizer.from_pretrained("./model")
        minimind_ids = minimind_tokenizer.encode(test_text)
        print(f"   - 输入: '{test_text}'")
        print(f"   - Token IDs: {minimind_ids}")
        print(f"   - Token数量: {len(minimind_ids)}")
        print(f"   - 解码: '{minimind_tokenizer.decode(minimind_ids)}'")
    except Exception as e:
        print(f"   - 失败: {e}")
    
    # 2. 使用GPT-2 tokenizer
    print("\n2. GPT-2 tokenizer:")
    try:
        gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        gpt2_ids = gpt2_tokenizer.encode(test_text)
        print(f"   - 输入: '{test_text}'")
        print(f"   - Token IDs: {gpt2_ids}")
        print(f"   - Token数量: {len(gpt2_ids)}")
        print(f"   - 解码: '{gpt2_tokenizer.decode(gpt2_ids)}'")
    except Exception as e:
        print(f"   - 失败: {e}")

def create_tokenizer_migration_guide():
    """创建tokenizer迁移指南"""
    
    print("\n=== Tokenizer迁移指南 ===\n")
    
    print("情况1: 使用一体化格式tokenizer（如MiniMind、Qwen）")
    print("  ✅ 只需要: tokenizer.json + tokenizer_config.json")
    print("  📁 文件结构:")
    print("    tokenizer/")
    print("    ├── tokenizer.json      # 核心文件（必须）")
    print("    └── tokenizer_config.json  # 配置文件（必须）")
    
    print("\n情况2: 使用传统格式tokenizer（如GPT-2、BERT）")
    print("  ✅ 需要: vocab.json + merges.txt + tokenizer_config.json")
    print("  📁 文件结构:")
    print("    tokenizer/")
    print("    ├── vocab.json          # 词汇表（必须）")
    print("    ├── merges.txt          # BPE规则（必须，如果使用BPE）")
    print("    ├── tokenizer_config.json  # 配置（必须）")
    print("    └── special_tokens_map.json  # 特殊token（可选）")
    
    print("\n关键检查点:")
    print("  1. 确认tokenizer类型（BPE、WordPiece、SentencePiece等）")
    print("  2. 检查词汇表大小是否与模型匹配")
    print("  3. 验证特殊token配置是否正确")
    print("  4. 测试tokenizer是否能正常编码/解码")

if __name__ == "__main__":
    # 运行分析
    analyze_minimind_tokenizer()
    analyze_qwen_tokenizer()
    analyze_gpt2_tokenizer()
    
    # 演示使用
    demonstrate_tokenizer_usage()
    
    # 创建迁移指南
    create_tokenizer_migration_guide()
    
    print("\n=== 总结 ===")
    print("✅ 一体化格式tokenizer: 只需要tokenizer.json + tokenizer_config.json")
    print("✅ 传统格式tokenizer: 需要vocab.json + merges.txt + tokenizer_config.json")
    print("🔍 关键: 检查tokenizer的实际文件结构，确保所有必要文件都存在")