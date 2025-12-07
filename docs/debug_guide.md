# MiniMind 虚拟环境调试指南

## 1. 环境激活

### 每次使用前激活环境
```bash
# 激活虚拟环境
source minimind_venv/bin/activate

# 验证环境
python --version  # 应该显示 Python 3.9.6
which python     # 应该指向虚拟环境中的Python
```

### 退出环境
```bash
deactivate
```

## 2. 基础调试方法

### 运行调试助手
```bash
source minimind_venv/bin/activate
python debug_helper.py
```

### 运行预训练调试
```bash
source minimind_venv/bin/activate
python debug_pretrain.py
```

### 运行正式训练
```bash
source minimind_venv/bin/activate
python trainer/train_pretrain.py
```

## 3. 高级调试技巧

### 使用Python调试器（pdb）

#### 方法1：命令行调试
```bash
# 使用pdb运行脚本
python -m pdb debug_pretrain.py

# pdb常用命令
# n(ext) - 执行下一行
# s(tep) - 进入函数
# c(ontinue) - 继续执行
# l(ist) - 显示代码
# p <变量> - 打印变量
# q(uit) - 退出
```

#### 方法2：在代码中插入断点
```python
# 在需要调试的地方插入
import pdb; pdb.set_trace()

# 或者使用breakpoint()（Python 3.7+）
breakpoint()
```

### 日志调试
```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 在代码中添加日志
logger = logging.getLogger(__name__)
logger.debug("调试信息: %s", variable)
logger.info("信息: %s", message)
```

## 4. 常见问题调试

### 内存问题调试
```python
import torch
import psutil

# 检查GPU内存
if torch.cuda.is_available():
    print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU内存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 检查系统内存
memory = psutil.virtual_memory()
print(f"系统内存使用: {memory.percent}%")
```

### 数据加载调试
```python
from dataset.lm_dataset import PretrainDataset
from transformers import AutoTokenizer

# 测试数据加载
tokenizer = AutoTokenizer.from_pretrained("./model")
dataset = PretrainDataset("dataset/mini_pretrain_data.jsonl", tokenizer)

# 检查数据样本
print("数据集长度:", len(dataset))
print("第一个样本:", dataset[0])
print("样本键:", list(dataset[0].keys()))
```

### 模型调试
```python
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

# 测试模型配置
config = MiniMindConfig()
print("模型配置:", config)

# 测试模型初始化
model = MiniMindForCausalLM(config)
print("模型参数量:", sum(p.numel() for p in model.parameters()))

# 测试前向传播
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 10))
outputs = model(input_ids)
print("输出形状:", outputs.logits.shape)
```

## 5. 性能优化调试

### 检查训练速度
```python
import time

# 计时训练循环
start_time = time.time()
# 训练代码...
end_time = time.time()
print(f"训练耗时: {end_time - start_time:.2f}秒")
```

### 检查数据加载速度
```python
from torch.utils.data import DataLoader

# 测试数据加载速度
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

start_time = time.time()
for i, batch in enumerate(dataloader):
    if i >= 10:  # 只测试10个batch
        break
end_time = time.time()
print(f"数据加载速度: {10/(end_time-start_time):.2f} batch/秒")
```

## 6. 错误处理调试

### 异常捕获
```python
try:
    # 可能出错的代码
    result = some_function()
except Exception as e:
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print(f"错误位置: {e.__traceback__.tb_lineno}")
```

### 梯度检查
```python
# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}梯度范数: {param.grad.norm().item():.6f}")
    else:
        print(f"{name}无梯度")
```

## 7. 实用调试脚本

### 快速环境检查
```bash
#!/bin/bash
# 保存为 check_env.sh

source minimind_venv/bin/activate

echo "=== 环境状态检查 ==="
python -c "
import torch, transformers, sys
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

echo "=== 文件检查 ==="
ls -la dataset/ | grep jsonl
ls -la model/ | grep -E '(json|bin|pth)'
```

## 8. IDE集成调试

### VS Code调试配置
创建 `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: MiniMind Debug",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            },
            "args": []
        }
    ]
}
```

## 9. 调试最佳实践

1. **从小开始**: 先用小数据集测试
2. **逐步调试**: 先测试数据加载，再测试模型
3. **记录日志**: 重要步骤添加日志
4. **版本控制**: 调试前后提交代码
5. **备份配置**: 修改重要配置前备份

记住：好的调试习惯能大大提高开发效率！