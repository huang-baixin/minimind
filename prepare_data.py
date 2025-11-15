#!/usr/bin/env python3
"""
MiniMind 数据准备脚本
创建预训练所需的示例数据集
"""

import json
from pathlib import Path


class MiniMindDataPreparer:
    """MiniMind 数据准备器"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def create_sample_pretrain_data(self):
        """创建示例预训练数据"""
        print("创建示例预训练数据...")
        
        # 创建多样化的中文文本样本
        sample_texts = []
        
        # 1. 维基百科风格文本
        wiki_samples = [
            "深度学习是机器学习的一个分支，它试图模拟人脑的工作方式。",
            "自然语言处理是人工智能领域的重要研究方向，涉及计算机与人类语言的交互。",
            "预训练语言模型在各种NLP任务中表现出色，如文本分类、问答和机器翻译。",
            "中文文本处理需要考虑分词、编码和语义理解等复杂问题。",
            "Transformer架构通过自注意力机制实现了高效的序列建模。",
            "PyTorch是一个开源的机器学习框架，广泛应用于深度学习研究和开发。",
            "神经网络通过多层非线性变换学习数据的复杂表示。",
            "反向传播算法是训练神经网络的核心技术之一。",
            "梯度下降优化算法用于最小化损失函数。",
            "过拟合是机器学习中常见的问题，需要通过正则化等技术来解决。"
        ]
        
        # 2. 新闻风格文本
        news_samples = [
            "近日，人工智能技术在医疗领域的应用取得了显著进展。",
            "研究人员开发出新的算法，能够更准确地预测疾病发展趋势。",
            "科技公司宣布推出新一代AI芯片，性能提升显著。",
            "教育部门推动人工智能课程进入中小学课堂。",
            "环保组织利用AI技术监测森林砍伐情况。",
            "金融行业采用机器学习算法进行风险评估和欺诈检测。",
            "自动驾驶技术在城市交通中的应用逐步扩大。",
            "智能家居设备通过语音识别技术提供更便捷的用户体验。",
            "农业领域引入无人机和AI技术提高生产效率。",
            "电子商务平台利用推荐算法提升用户购物体验。"
        ]
        
        # 3. 技术文档风格文本
        tech_samples = [
            "要安装PyTorch，可以使用pip命令：pip install torch torchvision。",
            "模型训练过程中需要设置合适的学习率和批次大小。",
            "数据预处理包括清洗、标准化和特征工程等步骤。",
            "模型评估指标包括准确率、精确率、召回率和F1分数。",
            "交叉验证技术有助于更可靠地评估模型性能。",
            "GPU加速可以显著提高深度学习模型的训练速度。",
            "迁移学习允许将预训练模型应用于新的任务。",
            "数据增强技术可以增加训练数据的多样性。",
            "模型部署需要考虑性能、可扩展性和安全性。",
            "持续集成和持续部署有助于自动化机器学习工作流。"
        ]
        
        # 4. 通用文本
        general_samples = [
            "春天来了，万物复苏，大地一片生机勃勃的景象。",
            "学习新知识需要耐心和坚持，不能急于求成。",
            "团队合作是现代工作中不可或缺的重要能力。",
            "健康的生活方式包括均衡饮食和适量运动。",
            "阅读是获取知识和提升自我的有效途径。",
            "时间管理对于提高工作效率至关重要。",
            "良好的沟通能力有助于建立和谐的人际关系。",
            "创新思维是推动社会进步的重要动力。",
            "环境保护需要每个人的参与和努力。",
            "传统文化是中华民族的宝贵精神财富。"
        ]
        
        # 组合所有样本并重复创建足够的数据量
        all_samples = wiki_samples + news_samples + tech_samples + general_samples
        
        # 重复样本创建10000条数据
        sample_texts = all_samples * 250  # 40 * 250 = 10000
        
        print(f"创建了 {len(sample_texts)} 条预训练样本")
        return sample_texts
    
    def process_pretrain_data(self):
        """处理预训练数据"""
        print("\n=== 处理预训练数据 ===")
        
        # 创建示例预训练数据
        all_texts = self.create_sample_pretrain_data()
        
        # 保存为JSONL格式
        output_file = self.data_dir / "pretrain_data.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                json.dump({"text": text}, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"预训练数据保存完成: {output_file.name} ({len(all_texts)} 条样本)")
    
    def create_mini_datasets(self):
        """创建小规模预训练数据集用于快速测试"""
        print("\n=== 创建小规模测试数据集 ===")
        
        datasets_to_minify = ["pretrain_data.jsonl"]
        
        for dataset_file in datasets_to_minify:
            full_path = self.data_dir / dataset_file
            mini_path = self.data_dir / f"mini_{dataset_file}"
            
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 取前1000行作为小数据集
                mini_lines = lines[:1000]
                
                with open(mini_path, 'w', encoding='utf-8') as f:
                    f.writelines(mini_lines)
                
                print(f"创建小规模数据集: {mini_path.name} ({len(mini_lines)} 条样本)")
    
    def prepare_pretrain_data(self):
        """准备预训练数据"""
        print("开始准备MiniMind项目预训练数据...")
        
        # 显示将要创建的数据集信息
        print(f"\n=== 预训练数据源 ===")
        print("  - 示例数据: 创建多样化的中文文本样本")
        print("  - 包含维基百科、新闻、技术文档和通用文本风格")
        
        # 处理预训练数据
        self.process_pretrain_data()
        
        # 创建小规模测试数据集
        self.create_mini_datasets()
        
        print("\n=== 预训练数据准备完成 ===")
        print("生成的数据文件:")
        for file in self.data_dir.glob("*pretrain*.jsonl"):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.1f} MB")
        
        print("\n下一步:")
        print("1. 运行数据验证: python validate_data.py")
        print("2. 运行分词器训练: python scripts/train_tokenizer.py")
        print("3. 开始预训练: python trainer/train_pretrain.py")


if __name__ == "__main__":
    preparer = MiniMindDataPreparer()
    preparer.prepare_pretrain_data()