#!/usr/bin/env python3
"""
梯度与权重Shape一致性演示
验证反向传播中梯度和权重维度匹配的数学原理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def demo_gradient_weight_shape_matching():
    """演示梯度与权重的shape匹配"""
    print("=" * 60)
    print("梯度与权重Shape一致性验证")
    print("=" * 60)
    
    # 1. 线性层梯度验证
    print("1. 线性层 (Linear Layer)")
    linear = nn.Linear(3, 5)  # 输入维度3, 输出维度5
    x = torch.randn(2, 3)     # batch_size=2, input_dim=3
    y_true = torch.randint(0, 5, (2,))
    
    print(f"   权重W形状: {linear.weight.shape}")  # [5, 3]
    print(f"   偏置b形状: {linear.bias.shape}")    # [5,]
    print(f"   输入x形状: {x.shape}")             # [2, 3]
    print(f"   输出y形状: {y_true.shape}")        # [2,]
    
    # 前向传播
    y_pred = linear(x)
    loss = F.cross_entropy(y_pred, y_true)
    
    # 反向传播
    loss.backward()
    
    print(f"   梯度dW形状: {linear.weight.grad.shape}")  # [5, 3] - 与W一致!
    print(f"   梯度db形状: {linear.bias.grad.shape}")    # [5,] - 与b一致!
    print(f"   ✓ 梯度与权重Shape完全匹配!")
    
    # 2. 卷积层梯度验证
    print("\n2. 卷积层 (Conv2d Layer)")
    conv = nn.Conv2d(1, 3, kernel_size=3)  # 输入通道1, 输出通道3
    x_img = torch.randn(1, 1, 5, 5)       # batch=1, channels=1, height=5, width=5
    y_target = torch.randint(0, 3, (1,))
    
    print(f"   权重kernel形状: {conv.weight.shape}")    # [3, 1, 3, 3]
    print(f"   偏置bias形状: {conv.bias.shape}")        # [3,]
    print(f"   输入图像形状: {x_img.shape}")            # [1, 1, 5, 5]
    
    # 前向传播
    conv_out = conv(x_img)
    loss = F.cross_entropy(conv_out.view(1, -1), y_target)
    
    # 反向传播
    loss.backward()
    
    print(f"   梯度dKernel形状: {conv.weight.grad.shape}")  # [3, 1, 3, 3] - 与kernel一致!
    print(f"   梯度dBias形状: {conv.bias.grad.shape}")      # [3,] - 与bias一致!
    print(f"   ✓ 卷积层梯度Shape也完全匹配!")
    
    # 3. Embedding层梯度验证
    print("\n3. Embedding层")
    vocab_size, embed_dim = 1000, 64
    embedding = nn.Embedding(vocab_size, embed_dim)
    input_ids = torch.randint(0, vocab_size, (4, 10))  # batch=4, seq_len=10
    
    print(f"   嵌入矩阵形状: {embedding.weight.shape}")     # [1000, 64]
    print(f"   输入序列形状: {input_ids.shape}")           # [4, 10]
    
    # 前向传播
    embedded = embedding(input_ids)
    loss = embedded.sum()  # 简单损失函数
    
    # 反向传播
    loss.backward()
    
    print(f"   梯度dEmbed形状: {embedding.weight.grad.shape}")  # [1000, 64] - 与嵌入矩阵一致!
    print(f"   ✓ Embedding层梯度Shape也完全匹配!")

def demo_mathematical_principle():
    """演示梯度与权重维度匹配的数学原理"""
    print("\n" + "=" * 60)
    print("数学原理: 为什么梯度与权重Shape必须一致")
    print("=" * 60)
    
    # 简单线性回归示例
    print("以线性回归为例: y = W·x + b")
    print()
    
    # 创建数据
    torch.manual_seed(42)
    W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5]])  # 真实权重 [2, 2]
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])        # 输入 [2, 2]
    y_true = torch.matmul(X, W_true.T) + torch.tensor([1.0, 2.0])  # [2, 2]
    
    # 定义模型
    W = torch.randn(2, 2, requires_grad=True)  # 模型权重
    b = torch.randn(2, requires_grad=True)     # 模型偏置
    
    print(f"真实权重W_true形状: {W_true.shape}")
    print(f"输入X形状: {X.shape}")
    print(f"模型权重W形状: {W.shape}")
    print(f"模型偏置b形状: {b.shape}")
    
    # 前向传播
    y_pred = torch.matmul(X, W.T) + b
    loss = F.mse_loss(y_pred, y_true)
    
    print(f"\n预测y_pred形状: {y_pred.shape}")
    print(f"真实值y_true形状: {y_true.shape}")
    print(f"损失loss形状: {loss.shape}")
    
    # 反向传播
    loss.backward()
    
    print(f"\n反向传播后:")
    print(f"梯度dW形状: {W.grad.shape}")
    print(f"梯度db形状: {b.grad.shape}")
    
    # 数学解释
    print(f"\n数学原理:")
    print(f"L = ||y_pred - y_true||² = ||X·W^T + b - y_true||²")
    print(f"∂L/∂W = 2·(X^T)·(y_pred - y_true) / n")
    print(f"∂L/∂b = 平均值(y_pred - y_true)")
    print()
    print(f"由于矩阵乘法运算:")
    print(f"X: [n, d_in] * W^T: [d_out, d_in] = y_pred: [n, d_out]")
    print(f"∂L/∂W: [d_out, d_in] 必须与 W: [d_out, d_in] 形状相同")
    print(f"这样才能进行: W = W - learning_rate * ∂L/∂W")

def demo_parameter_update():
    """演示参数更新的维度匹配"""
    print("\n" + "=" * 60)
    print("参数更新: 梯度与权重的维度匹配")
    print("=" * 60)
    
    # 创建简单模型
    model = nn.Linear(4, 2)
    
    # 模拟训练步骤
    x = torch.randn(3, 4)
    y = torch.randint(0, 2, (3,))
    
    # 前向传播
    output = model(x)
    loss = F.cross_entropy(output, y)
    
    # 反向传播
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    
    print("参数更新前:")
    print(f"权重W:\n{model.weight.data}")
    print(f"梯度dW:\n{model.weight.grad}")
    print(f"权重W形状: {model.weight.shape}")
    print(f"梯度dW形状: {model.weight.grad.shape}")
    
    # 验证形状匹配
    print(f"\n形状匹配检查:")
    print(f"W.shape == dW.shape: {model.weight.shape == model.weight.grad.shape}")
    print(f"W.shape[0] == dW.shape[0]: {model.weight.shape[0] == model.weight.grad.shape[0]}")
    print(f"W.shape[1] == dW.shape[1]: {model.weight.shape[1] == model.weight.grad.shape[1]}")
    
    # 执行更新
    optimizer.step()
    
    print(f"\n参数更新后:")
    print(f"权重W:\n{model.weight.data}")
    print("✓ 更新操作成功执行!")

def demo_why_shape_must_match():
    """演示为什么形状必须匹配"""
    print("\n" + "=" * 60)
    print("为什么梯度与权重Shape必须一致？")
    print("=" * 60)
    
    print("1. 数学必要性:")
    print("   - 梯度表示损失函数对每个参数的变化率")
    print("   - 如果权重是矩阵，梯度也必须是相同形状的矩阵")
    print("   - 这样才能进行逐元素的操作: W_new = W_old - lr * grad")
    
    print("\n2. 维度一致性原理:")
    print("   对于函数 f: R^(n×m) → R:")
    print("   - 输入: W ∈ R^(n×m) (n行m列的权重矩阵)")
    print("   - 输出: f(W) ∈ R (标量损失)")
    print("   - 梯度: ∇f(W) ∈ R^(n×m) (与W相同维度)")
    
    print("\n3. 反向传播的链式法则:")
    print("   ∂L/∂W[i,j] = ∂L/∂y · ∂y/∂W[i,j]")
    print("   其中∂y/∂W[i,j]是一个标量，所以∂L/∂W[i,j]也是标量")
    print("   对于每个(i,j)位置都有对应的梯度值")
    
    print("\n4. 参数更新公式:")
    print("   W_new = W_old - α · ∂L/∂W")
    print("   这个减法要求两个张量形状完全一致!")

def main():
    """主函数"""
    print("梯度与权重Shape一致性完整演示")
    print("作者: MiniMind Assistant")
    
    try:
        # 演示1: 基本shape匹配
        demo_gradient_weight_shape_matching()
        
        # 演示2: 数学原理
        demo_mathematical_principle()
        
        # 演示3: 参数更新
        demo_parameter_update()
        
        # 演示4: 必要性解释
        demo_why_shape_must_match()
        
        print("\n" + "=" * 60)
        print("总结: 梯度和权重的shape必然一致!")
        print("=" * 60)
        
        print("核心要点:")
        print("✓ 梯度是损失函数对每个参数的变化率")
        print("✓ 每个参数对应一个梯度值")
        print("✓ 参数更新的数学公式要求形状完全匹配")
        print("✓ 反向传播的链式法则保证维度一致性")
        print("✓ 这是深度学习的数学基础")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()