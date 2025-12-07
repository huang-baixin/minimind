#!/usr/bin/env python3
"""
Loss Backward Demo - 损失反向传播完整演示
展示从logits到loss再到backward()的完整过程
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SimpleModel(torch.nn.Module):
    """简单的线性模型用于演示"""
    def __init__(self, input_size=3, hidden_size=4, output_size=3):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

def demo_loss_calculation():
    """演示损失计算的详细过程"""
    print("=" * 60)
    print("1. 损失计算完整过程演示")
    print("=" * 60)
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 创建模拟数据
    batch_size = 2
    seq_length = 2
    vocab_size = 3
    
    # 模拟logits: [batch_size, seq_length, vocab_size]
    logits = torch.randn(batch_size, seq_length, vocab_size, requires_grad=True)
    targets = torch.tensor([2, 1, 0, 2])  # [batch_size * seq_length]
    
    print(f"原始logits形状: {logits.shape}")
    print(f"原始targets: {targets}")
    
    # 步骤1: 重塑logits
    logits_reshaped = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    
    print(f"\n重塑后logits形状: {logits_reshaped.shape}")
    print(f"重塑后targets形状: {targets_flat.shape}")
    
    # 步骤2: 计算softmax概率
    probs = F.softmax(logits_reshaped, dim=-1)
    print(f"\nSoftmax概率分布:")
    for i, prob in enumerate(probs):
        print(f"  位置{i}: {prob.tolist()}")
    
    # 步骤3: 计算交叉熵损失
    loss = F.cross_entropy(logits_reshaped, targets_flat)
    print(f"\n交叉熵损失: {loss}")
    print(f"损失类型: {type(loss)}")
    print(f"损失形状: {loss.shape}")
    print(f"损失维度: {loss.ndim}")
    
    # 步骤4: 手动验证损失计算
    print("\n手动验证损失计算:")
    log_probs = F.log_softmax(logits_reshaped, dim=-1)
    
    total_loss = 0
    for i, target in enumerate(targets_flat):
        target_log_prob = log_probs[i, target]
        print(f"  位置{target}的目标对数概率: {target_log_prob:.4f}")
        total_loss -= target_log_prob
    
    manual_loss = total_loss / len(targets_flat)
    print(f"手动计算的平均损失: {manual_loss:.4f}")
    
    return loss, logits

def demo_backward_process(loss, logits):
    """演示反向传播过程"""
    print("\n" + "=" * 60)
    print("2. 反向传播过程演示")
    print("=" * 60)
    
    # 检查梯度
    print(f"反向传播前logits梯度: {logits.grad}")
    
    # 执行反向传播
    loss.backward()
    
    print(f"反向传播后logits梯度形状: {logits.grad.shape}")
    print(f"反向传播后logits梯度:")
    print(logits.grad)
    
    # 验证梯度计算的链式法则
    print("\n梯度验证 (链式法则):")
    
    # 计算理论梯度: softmax - one_hot
    logits_reshaped = logits.view(-1, 3)
    targets = torch.tensor([2, 1, 0, 2])
    
    probs = F.softmax(logits_reshaped, dim=-1)
    one_hot = F.one_hot(targets, num_classes=3).float()
    theoretical_grad = probs - one_hot
    
    print("理论梯度 (softmax - one_hot):")
    print(theoretical_grad)
    
    # 验证是否匹配
    actual_grad = logits.grad.view(-1, 3)
    print(f"\n实际梯度:")
    print(actual_grad)
    
    print(f"\n梯度是否匹配: {torch.allclose(theoretical_grad, actual_grad, atol=1e-6)}")

def demo_scalar_tensor_properties():
    """演示标量tensor的性质"""
    print("\n" + "=" * 60)
    print("3. 标量tensor性质演示")
    print("=" * 60)
    
    # 创建标量tensor
    scalar_tensor = torch.tensor(3.14)
    print(f"标量tensor: {scalar_tensor}")
    print(f"形状: {scalar_tensor.shape}")
    print(f"维度: {scalar_tensor.ndim}")
    print(f"数据类型: {scalar_tensor.dtype}")
    
    # 与Python标量对比
    python_scalar = 3.14
    print(f"\nPython标量: {python_scalar}")
    print(f"类型: {type(python_scalar)}")
    
    # 标量tensor可以调用backward()
    print(f"\n标量tensor可调用backward(): {hasattr(scalar_tensor, 'backward')}")
    
    # 演示backward调用
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 2
    y.backward()
    
    print(f"y = x² 在 x=2 处的导数: {x.grad}")

def demo_computational_graph():
    """演示计算图的构建和反向传播"""
    print("\n" + "=" * 60)
    print("4. 计算图演示")
    print("=" * 60)
    
    # 手动构建一个简单的计算图
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    w = torch.tensor([0.5, 1.5], requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    
    # 前向传播
    y_pred = torch.dot(x, w) + b
    loss = (y_pred - 3.0) ** 2
    
    print(f"输入 x: {x}")
    print(f"权重 w: {w}")
    print(f"偏置 b: {b}")
    print(f"预测 y_pred: {y_pred}")
    print(f"真实值: 3.0")
    print(f"损失: {loss}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n梯度:")
    print(f"dL/dx: {x.grad}")
    print(f"dL/dw: {w.grad}")
    print(f"dL/db: {b.grad}")
    
    # 手动验证梯度
    # L = (y_pred - 3)² = (x·w + b - 3)²
    # dL/dw = 2(y_pred - 3) * x
    manual_grad_w = 2 * (y_pred - 3) * x
    print(f"\n手动计算 dL/dw: {manual_grad_w}")

def demo_mixed_precision_backward():
    """演示混合精度训练中的反向传播"""
    print("\n" + "=" * 60)
    print("5. 混合精度训练中的反向传播")
    print("=" * 60)
    
    # 模拟混合精度训练
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # 创建输入数据
    x = torch.randn(4, 3)
    targets = torch.randint(0, 3, (4,))
    
    print("混合精度训练步骤:")
    
    # 步骤1: 前向传播
    with torch.cuda.amp.autocast():
        outputs = model(x)
        loss = F.cross_entropy(outputs, targets)
    
    print(f"1. 原始损失: {loss}")
    print(f"   损失类型: {type(loss)}")
    print(f"   损失形状: {loss.shape}")
    
    # 步骤2: 清零梯度
    optimizer.zero_grad()
    print("2. 清零梯度")
    
    # 步骤3: 反向传播 (scaled)
    scaler.scale(loss).backward()
    print("3. 缩放后反向传播")
    
    # 步骤4: 优化器步骤
    scaler.step(optimizer)
    print("4. 优化器步骤")
    
    # 步骤5: 更新scaler
    scaler.update()
    print("5. 更新scaler")

def visualize_loss_computation():
    """可视化损失计算过程"""
    print("\n" + "=" * 60)
    print("6. 损失计算可视化")
    print("=" * 60)
    
    # 创建简单的2D损失曲面
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 模拟Rosenbrock函数
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲面
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, Z, levels=50)
    plt.colorbar(label='Loss')
    plt.title('Rosenbrock Loss Surface')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # 绘制梯度下降路径
    plt.subplot(1, 2, 2)
    plt.contour(X, Y, Z, levels=50, alpha=0.3)
    
    # 模拟梯度下降路径
    path_x = [-1.5, -1.2, -0.8, -0.4, 0.0, 0.3, 0.6, 0.9]
    path_y = [1.5, 1.2, 0.8, 0.4, 0.1, 0.2, 0.5, 0.8]
    plt.plot(path_x, path_y, 'ro-', label='Gradient Descent Path')
    plt.plot(1, 1, 'g*', markersize=15, label='Global Minimum')
    plt.colorbar(label='Loss')
    plt.title('Gradient Descent Optimization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/huangbaixin/project/minimind/loss_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为: loss_visualization.png")

def main():
    """主函数"""
    print("Loss Backward Demo - 损失反向传播完整演示")
    print("作者: MiniMind Assistant")
    print("日期: 2024")
    
    try:
        # 演示1: 损失计算过程
        loss, logits = demo_loss_calculation()
        
        # 演示2: 反向传播过程
        demo_backward_process(loss, logits)
        
        # 演示3: 标量tensor性质
        demo_scalar_tensor_properties()
        
        # 演示4: 计算图
        demo_computational_graph()
        
        # 演示5: 混合精度训练
        demo_mixed_precision_backward()
        
        # 演示6: 可视化
        visualize_loss_computation()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        print("\n关键要点总结:")
        print("1. loss是标量tensor (shape: torch.Size([]))")
        print("2. 标量tensor可以直接调用backward()")
        print("3. 反向传播使用链式法则计算梯度")
        print("4. 梯度储存在每个tensor的.grad属性中")
        print("5. 混合精度训练使用GradScaler处理梯度缩放")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()