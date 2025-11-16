# MiniMind 预训练指南

## 预训练概述

MiniMind 预训练过程使用标准的语言模型预训练方法，通过大规模文本数据训练模型学习语言表示。

## 预训练数据

项目提供了两个预训练数据集：

1. **mini_pretrain_data.jsonl** - 小型演示数据集，用于快速测试
2. **pretrain_data.jsonl** - 完整的预训练数据集

## 启动预训练

### 基本命令

```bash
# 使用默认配置开始预训练
cd /root/proj/minimind
python trainer/train_pretrain.py

# 使用自定义配置
python trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --max_seq_len 512 \
    --data_path dataset/pretrain_data.jsonl \
    --save_dir out \
    --save_weight pretrain
```

### 关键参数说明

- `--epochs`: 训练轮数（建议1轮zero或2-6轮充分训练）
- `--batch_size`: 批次大小（根据GPU显存调整）
- `--learning_rate`: 学习率（默认5e-4）
- `--max_seq_len`: 序列最大长度（默认512）
- `--data_path`: 训练数据路径
- `--save_dir`: 模型保存目录
- `--save_weight`: 权重文件前缀

## 模型配置

MiniMind 默认配置：
- **参数量**: 25.8M
- **隐藏层维度**: 512
- **隐藏层数量**: 8
- **注意力头数**: 8
- **词汇表大小**: 6400

## 训练流程

1. **环境初始化** - 设置分布式训练和随机种子
2. **数据加载** - 加载预训练数据集
3. **模型初始化** - 创建MiniMind模型
4. **优化器设置** - 使用AdamW优化器
5. **混合精度训练** - 支持bfloat16和float16
6. **梯度累积** - 默认8步梯度累积
7. **模型保存** - 定期保存检查点

## 训练监控

### 日志输出
训练过程中会显示：
- 当前epoch和step
- 损失值
- 学习率
- 预计剩余时间

### 检查点保存
- 每100步保存一次模型
- 保存到`out/`目录
- 文件格式：`pretrain_512.pth`

## 高级配置

### 使用MoE架构
```bash
python trainer/train_pretrain.py --use_moe 1
```

### 从检查点恢复训练
```bash
python trainer/train_pretrain.py --from_resume 1
```

### 使用WandB监控
```bash
python trainer/train_pretrain.py --use_wandb
```

## 快速开始

### 1. 测试预训练流程
```bash
# 使用小型数据集快速测试
python trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 8 \
    --data_path dataset/mini_pretrain_data.jsonl \
    --save_dir test_out
```

### 2. 完整预训练
```bash
# 使用完整数据集训练
python trainer/train_pretrain.py \
    --epochs 3 \
    --batch_size 32 \
    --data_path dataset/pretrain_data.jsonl \
    --save_dir out
```

## 注意事项

1. **GPU显存**: 根据显存大小调整batch_size
2. **训练时间**: 完整预训练可能需要数小时到数天
3. **数据质量**: 确保预训练数据质量高
4. **监控训练**: 定期检查损失曲线避免过拟合

## 故障排除

### 常见问题

1. **显存不足**: 减小batch_size或使用梯度累积
2. **训练不稳定**: 调整学习率或使用学习率调度
3. **数据加载慢**: 检查数据文件路径和格式

### 性能优化

- 使用多GPU训练加速
- 启用混合精度训练
- 优化数据加载器配置

## 下一步

预训练完成后，可以进行：
1. **SFT微调** - 指令微调
2. **DPO训练** - 偏好对齐
3. **模型评估** - 性能测试

---

开始你的MiniMind预训练之旅吧！