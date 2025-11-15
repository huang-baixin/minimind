#!/usr/bin/env python3
"""
MiniMind 数据验证脚本
用于验证预训练数据的格式和质量
"""

import json
import os
from pathlib import Path
from collections import Counter
import re


class MiniMindDataValidator:
    """MiniMind 数据验证器"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = Path(data_dir)
        
    def validate_pretrain_data(self, file_path="pretrain_data.jsonl"):
        """验证预训练数据格式"""
        full_path = self.data_dir / file_path
        
        if not full_path.exists():
            print(f"❌ 预训练数据文件不存在: {full_path}")
            return False
        
        print(f"\n=== 验证预训练数据: {file_path} ===")
        
        issues = []
        stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "text_lengths": [],
            "char_counts": Counter(),
            "word_counts": Counter()
        }
        
        with open(full_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                stats["total_samples"] += 1
                
                try:
                    # 解析JSON
                    data = json.loads(line.strip())
                    
                    # 检查必需字段
                    if "text" not in data:
                        issues.append(f"第{line_num}行: 缺少 'text' 字段")
                        continue
                    
                    text = data["text"]
                    
                    # 检查文本类型
                    if not isinstance(text, str):
                        issues.append(f"第{line_num}行: 'text' 字段不是字符串类型")
                        continue
                    
                    # 检查文本长度
                    if len(text.strip()) == 0:
                        issues.append(f"第{line_num}行: 文本为空")
                        continue
                    
                    # 统计信息
                    stats["valid_samples"] += 1
                    stats["text_lengths"].append(len(text))
                    
                    # 字符统计
                    stats["char_counts"].update(text)
                    
                    # 词频统计（简单分词）
                    words = re.findall(r'\w+', text)
                    stats["word_counts"].update(words)
                    
                except json.JSONDecodeError as e:
                    issues.append(f"第{line_num}行: JSON解析错误 - {e}")
                except Exception as e:
                    issues.append(f"第{line_num}行: 未知错误 - {e}")
        
        # 输出验证结果
        print(f"📊 数据统计:")
        print(f"  总样本数: {stats['total_samples']}")
        print(f"  有效样本数: {stats['valid_samples']}")
        print(f"  数据质量: {stats['valid_samples']/stats['total_samples']*100:.1f}%")
        
        if stats['valid_samples'] > 0:
            print(f"\n📏 文本长度统计:")
            print(f"  平均长度: {sum(stats['text_lengths'])/len(stats['text_lengths']):.0f} 字符")
            print(f"  最小长度: {min(stats['text_lengths'])} 字符")
            print(f"  最大长度: {max(stats['text_lengths'])} 字符")
            
            print(f"\n🔤 字符统计 (前10):")
            for char, count in stats['char_counts'].most_common(10):
                print(f"  '{char}': {count} 次")
            
            print(f"\n📝 词频统计 (前10):")
            for word, count in stats['word_counts'].most_common(10):
                print(f"  '{word}': {count} 次")
        
        # 输出问题
        if issues:
            print(f"\n⚠️  发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... 还有 {len(issues)-10} 个问题未显示")
        else:
            print(f"\n✅ 数据格式验证通过!")
        
        return len(issues) == 0
    
    def validate_mini_dataset(self):
        """验证小规模数据集"""
        mini_path = self.data_dir / "mini_pretrain_data.jsonl"
        
        if not mini_path.exists():
            print(f"❌ 小规模数据集不存在: {mini_path}")
            return False
        
        print(f"\n=== 验证小规模数据集 ===")
        return self.validate_pretrain_data("mini_pretrain_data.jsonl")
    
    def check_data_files(self):
        """检查数据文件是否存在"""
        print("\n=== 检查数据文件 ===")
        
        files_to_check = [
            "pretrain_data.jsonl",
            "mini_pretrain_data.jsonl"
        ]
        
        all_exist = True
        for file_name in files_to_check:
            file_path = self.data_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"✅ {file_name}: {size_mb:.1f} MB")
            else:
                print(f"❌ {file_name}: 文件不存在")
                all_exist = False
        
        return all_exist
    
    def run_full_validation(self):
        """运行完整的数据验证"""
        print("🚀 开始MiniMind数据验证...")
        
        # 检查文件存在性
        files_ok = self.check_data_files()
        
        # 验证主数据集
        pretrain_ok = self.validate_pretrain_data()
        
        # 验证小数据集
        mini_ok = self.validate_mini_dataset()
        
        # 总结
        print(f"\n📋 验证总结:")
        print(f"  文件检查: {'✅ 通过' if files_ok else '❌ 失败'}")
        print(f"  预训练数据: {'✅ 通过' if pretrain_ok else '❌ 失败'}")
        print(f"  小规模数据: {'✅ 通过' if mini_ok else '❌ 失败'}")
        
        overall_success = files_ok and pretrain_ok and mini_ok
        if overall_success:
            print(f"\n🎉 所有验证通过! 数据准备就绪。")
        else:
            print(f"\n⚠️  部分验证失败，请检查数据文件。")
        
        return overall_success


def main():
    """主函数"""
    validator = MiniMindDataValidator()
    
    # 运行完整验证
    success = validator.run_full_validation()
    
    if success:
        print("\n✅ 数据验证完成，可以开始训练!")
        print("下一步:")
        print("1. 运行分词器训练: python scripts/train_tokenizer.py")
        print("2. 开始预训练: python trainer/train_pretrain.py")
    else:
        print("\n❌ 数据验证失败，请重新准备数据。")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())