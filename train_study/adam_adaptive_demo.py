#!/usr/bin/env python3
"""
Adam 自适应学习率演示
展示Adam如何为不同参数自动调整学习率
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AdaptiveLearningModel(nn.Module):
    """演示自适应学习率的模型 - Rosenbrock函数"""
    def __init__(self):
        super().__init__()
        # 初始化参数在不同区域
        self.x = nn.Parameter(torch.tensor(1.5, requires_grad=True))
        self.y = nn.Parameter(torch.tensor(-1.5, requires_grad=True))
    
    def forward(self):
        # Rosenbrock函数 - 著名的优化测试函数
        # 最小值在(1, 1)，函数值为0
        return (1 - self.x)**2 + 100 * (self.y - self.x**2)**2

class AdamLearningRateTracker:
    """跟踪Adam优化器中每个参数的学习率变化"""
    
    def __init__(self, model: nn.Module):
        self.original_params = {}
        self.lr_changes = {name: [] for name, _ in model.named_parameters()}
        
    def before_step(self, optimizer: optim.Adam):
        """在optimizer.step()之前记录状态"""
        for name, param in optimizer.model.named_parameters():
            if name in self.lr_changes:
                # 记录当前参数值
                self.lr_changes[name].append(param.data.clone())
    
    def track_lr_changes(self, model: nn.Module, optimizer: optim.Adam):
        """跟踪学习率变化"""
        for name, param in model.named_parameters():
            if name in self.lr_changes:
                # 这里我们模拟学习率的变化
                # 在实际中，学习率 = base_lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                if hasattr(optimizer, 'state') and param in optimizer.state:
                    state = optimizer.state[param]
                    if 'exp_avg' in state and 'exp_avg_sq' in state:
                        # 计算自适应学习率
                        t = optimizer.state[param].get('step', 1)
                        lr = optimizer.defaults['lr']
                        beta1, beta2 = optimizer.defaults['betas']
                        
                        bias_correction1 = 1 - beta1 ** t
                        bias_correction2 = 1 - beta2 ** t
                        
                        adaptive_lr = lr * (bias_correction1 / bias_correction2) ** 0.5
                        self.lr_changes[name].append(adaptive_lr)

def train_and_track(model: nn.Module, optimizer: optim.Adam, 
                   epochs: int = 200, track_steps: List[int] = None) -> dict:
    """训练模型并跟踪参数变化"""
    
    if track_steps is None:
        track_steps = [0, 10, 50, 100, 150, 199]
    
    trajectory = {'x': [], 'y': [], 'loss': [], 'epoch': []}
    
    for epoch in range(epochs):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        loss = model()
        
        # 反向传播
        loss.backward()
        
        # 记录轨迹
        trajectory['x'].append(model.x.item())
        trajectory['y'].append(model.y.item())
        trajectory['loss'].append(loss.item())
        trajectory['epoch'].append(epoch)
        
        # 更新参数
        optimizer.step()
        
        # 在指定步骤打印进度
        if epoch in track_steps:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.8f}, "
                  f"x = {model.x.item():.6f}, y = {model.y.item():.6f}")
    
    return trajectory

def visualize_adaptive_learning(true_minimum=(1, 1)):
    """可视化Adam的自适应学习率特性"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Adam 自适应学习率特性演示', fontsize=16, fontweight='bold')
    
    # Rosenbrock函数的可视化
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    # 1. 损失函数等高线 + SGD轨迹
    ax1 = axes[0, 0]
    contour = ax1.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # SGD轨迹
    model_sgd = AdaptiveLearningModel()
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.002)  # 较小学习率避免震荡
    sgd_traj = train_and_track(model_sgd, optimizer_sgd)
    
    ax1.plot(sgd_traj['x'], sgd_traj['y'], 'r-o', alpha=0.8, linewidth=2, 
             markersize=4, label='SGD轨迹')
    ax1.plot(true_minimum[0], true_minimum[1], 'g*', markersize=15, 
             label='真实最小值(1,1)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('SGD优化轨迹 (固定学习率)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 损失函数等高线 + Adam轨迹
    ax2 = axes[0, 1]
    ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), alpha=0.6)
    
    # Adam轨迹
    model_adam = AdaptiveLearningModel()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.002)
    adam_traj = train_and_track(model_adam, optimizer_adam)
    
    ax2.plot(adam_traj['x'], adam_traj['y'], 'b-o', alpha=0.8, linewidth=2, 
             markersize=4, label='Adam轨迹')
    ax2.plot(true_minimum[0], true_minimum[1], 'g*', markersize=15, 
             label='真实最小值(1,1)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Adam优化轨迹 (自适应学习率)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失函数下降对比
    ax3 = axes[0, 2]
    ax3.semilogy(sgd_traj['loss'], 'r-', label='SGD', linewidth=2)
    ax3.semilogy(adam_traj['loss'], 'b-', label='Adam', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('损失函数下降对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数x的变化轨迹
    ax4 = axes[1, 0]
    ax4.plot(sgd_traj['epoch'], sgd_traj['x'], 'r-', label='SGD', linewidth=2)
    ax4.plot(adam_traj['epoch'], adam_traj['x'], 'b-', label='Adam', linewidth=2)
    ax4.axhline(y=true_minimum[0], color='green', linestyle='--', 
                label=f'真实值: {true_minimum[0]}')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('参数 x')
    ax4.set_title('参数 x 的优化轨迹')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 参数y的变化轨迹
    ax5 = axes[1, 1]
    ax5.plot(sgd_traj['epoch'], sgd_traj['y'], 'r-', label='SGD', linewidth=2)
    ax5.plot(adam_traj['epoch'], adam_traj['y'], 'b-', label='Adam', linewidth=2)
    ax5.axhline(y=true_minimum[1], color='green', linestyle='--', 
                label=f'真实值: {true_minimum[1]}')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('参数 y')
    ax5.set_title('参数 y 的优化轨迹')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Adam vs SGD步数统计
    ax6 = axes[1, 2]
    sgd_steps = len([i for i in range(len(sgd_traj['loss'])) 
                     if sgd_traj['loss'][i] < 0.1])
    adam_steps = len([i for i in range(len(adam_traj['loss'])) 
                      if adam_traj['loss'][i] < 0.1])
    
    methods = ['SGD', 'Adam']
    steps_needed = [sgd_steps, adam_steps]
    colors = ['red', 'blue']
    
    bars = ax6.bar(methods, steps_needed, color=colors, alpha=0.7)
    ax6.set_ylabel('达到损失<0.1的步数')
    ax6.set_title('收敛速度对比')
    ax6.grid(True, alpha=0.3)
    
    # 在柱状图上添加数值
    for bar, step in zip(bars, steps_needed):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{step}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/huangbaixin/project/minimind/adam_adaptive_demo.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def explain_adam_algorithm():
    """解释Adam算法的工作原理"""
    
    print("\n" + "=" * 80)
    print("                   Adam算法原理深度解析")
    print("=" * 80)
    
    print("""
🔍 Rosenbrock函数特点:
   • 函数形状: 像一个弯曲的山谷
   • 最小值: (1, 1)，函数值 = 0
   • 挑战: 在x=1附近很平坦，在x=1附近很陡峭
   
🎯 SGD的挑战:
   • 固定学习率: 在平坦区域收敛慢，在陡峭区域可能震荡
   • 需要手动调整学习率调度
   
🚀 Adam的优势:
   • 自适应学习率: 根据梯度历史自动调整每个参数的学习率
   • 动量机制: 积累历史梯度方向，加速收敛
   • 偏差修正: 避免初期估计偏差
    """)
    
    print("\n📊 数学原理:")
    print("""
   1. 一阶矩估计 (动量):
      m_t = β₁ * m_{t-1} + (1 - β₁) * ∇θ_t
      
   2. 二阶矩估计 (自适应学习率):
      v_t = β₂ * v_{t-1} + (1 - β₂) * (∇θ_t)²
      
   3. 偏差修正:
      m̂_t = m_t / (1 - β₁^t)
      v̂_t = v_t / (1 - β₂^t)
      
   4. 参数更新:
      θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
    """)
    
    print("\n💡 实际效果:")
    print("   ✅ 在平坦区域: 增加学习率，加速收敛")
    print("   ✅ 在陡峭区域: 减小学习率，避免震荡")
    print("   ✅ 自动适应: 不需要手动调整学习率")

def main():
    """主函数"""
    
    explain_adam_algorithm()
    
    print("\n" + "=" * 80)
    print("           开始Adam自适应学习率演示")
    print("=" * 80)
    
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 训练轨迹对比
    print("\n🎯 训练SGD (固定学习率):")
    model_sgd = AdaptiveLearningModel()
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.002)
    sgd_traj = train_and_track(model_sgd, optimizer_sgd)
    
    print(f"\n   SGD最终结果:")
    print(f"   - 最终损失: {sgd_traj['loss'][-1]:.8f}")
    print(f"   - 参数x: {sgd_traj['x'][-1]:.6f} (真实值: 1.000)")
    print(f"   - 参数y: {sgd_traj['y'][-1]:.6f} (真实值: 1.000)")
    
    print("\n🚀 训练Adam (自适应学习率):")
    torch.manual_seed(42)  # 重置随机种子
    model_adam = AdaptiveLearningModel()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.002)
    adam_traj = train_and_track(model_adam, optimizer_adam)
    
    print(f"\n   Adam最终结果:")
    print(f"   - 最终损失: {adam_traj['loss'][-1]:.8f}")
    print(f"   - 参数x: {adam_traj['x'][-1]:.6f} (真实值: 1.000)")
    print(f"   - 参数y: {adam_traj['y'][-1]:.6f} (真实值: 1.000)")
    
    # 性能对比
    print(f"\n📊 性能对比:")
    sgd_final_loss = sgd_traj['loss'][-1]
    adam_final_loss = adam_traj['loss'][-1]
    
    sgd_steps_to_01 = next((i for i, loss in enumerate(sgd_traj['loss']) if loss < 0.1), len(sgd_traj['loss']))
    adam_steps_to_01 = next((i for i, loss in enumerate(adam_traj['loss']) if loss < 0.1), len(adam_traj['loss']))
    
    print(f"   - SGD达到损失<0.1的步数: {sgd_steps_to_01}")
    print(f"   - Adam达到损失<0.1的步数: {adam_steps_to_01}")
    print(f"   - Adam比SGD快: {sgd_steps_to_01 - adam_steps_to_01} 步")
    
    if adam_final_loss < sgd_final_loss:
        improvement = ((sgd_final_loss - adam_final_loss) / sgd_final_loss) * 100
        print(f"   - Adam最终损失比SGD好: {improvement:.2f}%")
    
    print(f"\n📈 生成详细可视化...")
    visualize_adaptive_learning()
    print(f"   📊 可视化图表已保存: adam_adaptive_demo.png")
    
    print(f"\n" + "=" * 80)
    print("           Adam自适应学习率演示完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()