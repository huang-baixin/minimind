#!/usr/bin/env python3
"""
Adam 优化器演示
对比 SGD 和 Adam 在优化问题上的表现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleLinearModel(nn.Module):
    """简单的线性回归模型"""
    def __init__(self):
        super().__init__()
        # 定义参数 w = 2.5, b = 1.3
        self.w = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
    
    def forward(self, x):
        return self.w * x + self.b

def generate_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成带噪声的训练数据
    真实函数: y = 2.5x + 1.3 + noise
    """
    torch.manual_seed(42)
    x = torch.linspace(-2, 2, n_samples)
    # 真实参数
    true_w, true_b = 2.5, 1.3
    y = true_w * x + true_b + 0.3 * torch.randn(n_samples)
    return x, y

def train_with_optimizer(model: nn.Module, 
                        optimizer: optim.Optimizer,
                        x: torch.Tensor, 
                        y: torch.Tensor, 
                        epochs: int = 100) -> dict:
    """训练模型并记录过程"""
    
    losses = []
    w_values = []
    b_values = []
    
    for epoch in range(epochs):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        y_pred = model(x)
        loss = nn.MSELoss()(y_pred.squeeze(), y)
        
        # 反向传播
        loss.backward()
        
        # 记录参数值
        w_values.append(model.w.item())
        b_values.append(model.b.item())
        losses.append(loss.item())
        
        # 更新参数
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, "
                  f"w = {model.w.item():.3f}, b = {model.b.item():.3f}")
    
    return {
        'losses': losses,
        'w_values': w_values,
        'b_values': b_values
    }

def visualize_training_process(true_w: float, true_b: float, 
                             sgd_results: dict, adam_results: dict,
                             x: torch.Tensor, y: torch.Tensor):
    """可视化训练过程"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SGD vs Adam 优化器对比', fontsize=16, fontweight='bold')
    
    # 1. 损失函数变化
    axes[0, 0].plot(sgd_results['losses'], label='SGD', color='red', alpha=0.7)
    axes[0, 0].plot(adam_results['losses'], label='Adam', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('损失函数变化')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 参数 w 的变化
    axes[0, 1].axhline(y=true_w, color='green', linestyle='--', label='真实值')
    axes[0, 1].plot(sgd_results['w_values'], label='SGD', color='red', alpha=0.7)
    axes[0, 1].plot(adam_results['w_values'], label='Adam', color='blue', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('参数 w')
    axes[0, 1].set_title('参数 w 的变化')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 参数 b 的变化
    axes[0, 2].axhline(y=true_b, color='green', linestyle='--', label='真实值')
    axes[0, 2].plot(sgd_results['b_values'], label='SGD', color='red', alpha=0.7)
    axes[0, 2].plot(adam_results['b_values'], label='Adam', color='blue', alpha=0.7)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('参数 b')
    axes[0, 2].set_title('参数 b 的变化')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. SGD 最终拟合效果
    final_sgd_w = sgd_results['w_values'][-1]
    final_sgd_b = sgd_results['b_values'][-1]
    
    axes[1, 0].scatter(x.numpy(), y.numpy(), alpha=0.6, s=20, label='数据点')
    x_line = torch.linspace(x.min(), x.max(), 100)
    axes[1, 0].plot(x_line.numpy(), (true_w * x_line + true_b).numpy(), 
                    'g--', label=f'真实函数 (w={true_w}, b={true_b})', linewidth=2)
    axes[1, 0].plot(x_line.numpy(), (final_sgd_w * x_line + final_sgd_b).numpy(), 
                    'r-', label=f'SGD拟合 (w={final_sgd_w:.2f}, b={final_sgd_b:.2f})', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_title('SGD 最终拟合效果')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Adam 最终拟合效果
    final_adam_w = adam_results['w_values'][-1]
    final_adam_b = adam_results['b_values'][-1]
    
    axes[1, 1].scatter(x.numpy(), y.numpy(), alpha=0.6, s=20, label='数据点')
    axes[1, 1].plot(x_line.numpy(), (true_w * x_line + true_b).numpy(), 
                    'g--', label=f'真实函数 (w={true_w}, b={true_b})', linewidth=2)
    axes[1, 1].plot(x_line.numpy(), (final_adam_w * x_line + final_adam_b).numpy(), 
                    'b-', label=f'Adam拟合 (w={final_adam_w:.2f}, b={final_adam_b:.2f})', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_title('Adam 最终拟合效果')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 参数轨迹对比 (w vs b)
    axes[1, 2].scatter(sgd_results['w_values'], sgd_results['b_values'], 
                       c='red', alpha=0.7, s=20, label='SGD轨迹', marker='o')
    axes[1, 2].scatter(adam_results['w_values'], adam_results['b_values'], 
                       c='blue', alpha=0.7, s=20, label='Adam轨迹', marker='s')
    axes[1, 2].scatter(true_w, true_b, c='green', s=100, marker='*', 
                       label='真实参数', edgecolor='black', linewidth=2)
    axes[1, 2].set_xlabel('参数 w')
    axes[1, 2].set_ylabel('参数 b')
    axes[1, 2].set_title('参数优化轨迹')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/huangbaixin/project/minimind/adam_vs_sgd_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数：演示Adam vs SGD"""
    
    print("=" * 60)
    print("           Adam 优化器演示")
    print("=" * 60)
    
    # 生成训练数据
    print("\n1. 生成训练数据...")
    x, y = generate_data(n_samples=100)
    print(f"   数据点数量: {len(x)}")
    print(f"   x 范围: [{x.min():.2f}, {x.max():.2f}]")
    print(f"   y 范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 设置训练参数
    epochs = 100
    learning_rate = 0.1
    
    print(f"\n2. 开始训练 (epochs={epochs}, lr={learning_rate})")
    print("-" * 40)
    
    # 训练 SGD
    print("\n🎯 训练 SGD 优化器:")
    model_sgd = SimpleLinearModel()
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)
    sgd_results = train_with_optimizer(model_sgd, optimizer_sgd, x, y, epochs)
    
    print(f"\n   SGD 最终结果:")
    print(f"   - 最终损失: {sgd_results['losses'][-1]:.6f}")
    print(f"   - 学习到的 w: {sgd_results['w_values'][-1]:.4f} (真实值: 2.5)")
    print(f"   - 学习到的 b: {sgd_results['b_values'][-1]:.4f} (真实值: 1.3)")
    
    # 训练 Adam
    print("\n🚀 训练 Adam 优化器:")
    model_adam = SimpleLinearModel()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
    adam_results = train_with_optimizer(model_adam, optimizer_adam, x, y, epochs)
    
    print(f"\n   Adam 最终结果:")
    print(f"   - 最终损失: {adam_results['losses'][-1]:.6f}")
    print(f"   - 学习到的 w: {adam_results['w_values'][-1]:.4f} (真实值: 2.5)")
    print(f"   - 学习到的 b: {adam_results['b_values'][-1]:.4f} (真实值: 1.3)")
    
    # 对比分析
    print(f"\n3. 对比分析:")
    print("-" * 40)
    sgd_final_loss = sgd_results['losses'][-1]
    adam_final_loss = adam_results['losses'][-1]
    
    print(f"   SGD 最终损失: {sgd_final_loss:.6f}")
    print(f"   Adam 最终损失: {adam_final_loss:.6f}")
    
    if adam_final_loss < sgd_final_loss:
        print(f"   🏆 Adam 比 SGD 好 {((sgd_final_loss - adam_final_loss) / sgd_final_loss * 100):.2f}%")
    else:
        print(f"   🏆 SGD 比 Adam 好 {((adam_final_loss - sgd_final_loss) / adam_final_loss * 100):.2f}%")
    
    # Adam 的核心优势
    print(f"\n4. Adam 的核心优势:")
    print("   ✨ 自适应学习率 - 为每个参数调整学习率")
    print("   ✨ 动量机制 - 减少震荡，加速收敛")
    print("   ✨ 偏差修正 - 避免初始阶段的不稳定")
    print("   ✨ 内存效率 - 只存储一阶和二阶矩估计")
    
    # 可视化结果
    print(f"\n5. 生成可视化图表...")
    visualize_training_process(2.5, 1.3, sgd_results, adam_results, x, y)
    print(f"   📊 图表已保存到: adam_vs_sgd_comparison.png")
    
    print(f"\n" + "=" * 60)
    print("           演示完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()