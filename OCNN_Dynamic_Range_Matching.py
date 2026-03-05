import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
from sklearn.metrics import confusion_matrix
import matplotlib
import argparse

# 导入硬件查找表
from LUT import lut as hardware_lut

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. 模型定义 (保持不变) ---
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_data_loaders(dataset_name='mnist', batch_size=64):
    if dataset_name.lower() == 'fashion_mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=1000,
                                                                                      shuffle=False)


# --- 2. 训练与评估函数 ---
def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, num_classes=10).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_onehot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, device, test_loader, compute_details=False):
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            if compute_details:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    acc = correct / len(test_loader.dataset)
    if not compute_details: return acc, None, None
    return acc, confusion_matrix(all_targets, all_preds), None


# --- 3. 核心改进：动态范围匹配与量化 ---

def analyze_and_quantize_layer(name, weight_tensor, lut):
    """
    对单个层的权重进行分析、缩放和量化
    """
    w_np = weight_tensor.cpu().detach().numpy()

    # 1. 动态范围分析
    w_max = np.max(np.abs(w_np))
    lut_max = np.max(np.abs(lut))

    # 2. 计算缩放因子 (Scaling Factor)
    # alpha是将软件权重映射到硬件LUT范围的系数
    # W_software / alpha = W_hardware (ideal)
    alpha = w_max / lut_max

    print(f"Layer {name}: Weight Range [{-w_max:.4f}, {w_max:.4f}] -> Scale Alpha: {alpha:.4f}")

    # 3. 归一化权重 (映射到 LUT 的物理范围 [-1, 1] 附近)
    w_normalized = w_np / alpha

    # 4. 最近邻量化 (Nearest Neighbor Quantization)
    # 利用广播机制查找最近的LUT值
    diff = np.abs(w_normalized[..., None] - lut[None, :])
    indices = np.argmin(diff, axis=-1)
    w_quantized_normalized = lut[indices]

    # 5. 恢复幅值 (De-normalization)
    # 在仿真中，我们要模拟“硬件权重 x 增益”的效果
    # w_simulated = w_hardware_state * alpha
    w_simulated = w_quantized_normalized * alpha

    return torch.from_numpy(w_simulated).to(weight_tensor.device), alpha, w_np.flatten(), w_simulated.flatten()


def apply_dynamic_range_quantization(model, lut, output_dir):
    quantized_model = copy.deepcopy(model)
    stats = {}

    print("\n--- Starting Dynamic Range Matching & Quantization ---")
    print(f"Hardware LUT Range: [{lut.min():.4f}, {lut.max():.4f}] (Levels: {len(lut)})")

    for name, module in quantized_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 量化权重
            new_weight, alpha, orig_w, quant_w = analyze_and_quantize_layer(
                name, module.weight.data, lut
            )
            module.weight.data = new_weight
            stats[name] = {'alpha': alpha, 'orig': orig_w, 'quant': quant_w}

    # 可视化：绘制分布匹配图 (Distribution Overlay)
    plot_distribution_overlay(stats, lut, output_dir)

    print("Quantization complete.")
    return quantized_model


# --- 4. 可视化模块 (新增分布图) ---

def plot_distribution_overlay(stats, lut, output_dir):
    """
    绘制权重分布与LUT物理状态的叠加图。
    这是论文中展示“软硬协同设计”有效性的关键图表。
    """
    # 选择一个代表性层（例如全连接层 fc1）进行绘图
    target_layer = 'fc1'
    if target_layer not in stats: return  # 防御性编程

    data = stats[target_layer]
    orig_weights = data['orig']
    scale = data['alpha']

    # 将LUT扩展到软件权重的尺度，以便在同一坐标系下对比
    # 这展示了：经过缩放后，这128个物理状态在软件权重的动态范围内落在哪里
    scaled_lut = lut * scale

    plt.figure(figsize=(10, 6), dpi=300)

    # 1. 绘制软件权重的直方图 (背景)
    sns.histplot(orig_weights, bins=100, color='skyblue', label='Software Weights (Float64)', kde=True, stat="density",
                 alpha=0.4)

    # 2. 绘制硬件LUT的离散点 (前景竖线)
    # 我们画出缩放后的LUT位置，展示它们如何“切割”权重分布
    for val in scaled_lut:
        plt.axvline(val, color='red', linestyle='-', alpha=0.3, linewidth=0.8)

    # 为了图例整洁，只给第一条线加标签
    plt.axvline(scaled_lut[0], color='red', linestyle='-', alpha=0.8, linewidth=1.5,
                label='Mapped Hardware States (128 Levels)')

    plt.title(f'Dynamic Range Matching: Layer {target_layer}', fontsize=14)
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(orig_weights.min() * 1.1, orig_weights.max() * 1.1)

    save_path = os.path.join(output_dir, 'distribution_matching.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Distribution matching plot saved to {save_path}")


def generate_report(output_dir, acc_fp, acc_q, dataset_name):
    with open(os.path.join(output_dir, 'final_report.txt'), 'w') as f:
        f.write(f"--- Hardware-Aware Simulation Report ({dataset_name}) ---\n")
        f.write(f"Full Precision Accuracy: {acc_fp * 100:.2f}%\n")
        f.write(f"Hardware-Mapped Accuracy: {acc_q * 100:.2f}%\n")
        f.write(f"Accuracy Drop: {(acc_fp - acc_q) * 100:.2f}%\n")
        f.write("\nKey Technique: Layer-wise Dynamic Range Matching\n")
        f.write("Status: Weights were scaled to maximally utilize the 128 non-uniform PCM states.")


# --- 5. 主程序 ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()

    OUTPUT_DIR = f'results_{args.dataset}_dynamic'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据与模型
    train_loader, test_loader = get_data_loaders(args.dataset)
    model = DNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 2. 训练全精度模型
    print("--- Training Full Precision Model ---")
    for epoch in range(5):  # 演示用5轮，实际建议10+
        loss = train_epoch(model, device, train_loader, optimizer, criterion)
        print(f"Epoch {epoch + 1}: Loss {loss:.4f}")

    acc_fp, cm_fp, _ = evaluate(model, device, test_loader, True)
    print(f"Full Precision Acc: {acc_fp * 100:.2f}%")

    # 3. 动态范围匹配与量化
    final_lut = hardware_lut.astype(np.float32)
    quantized_model = apply_dynamic_range_quantization(model, final_lut, OUTPUT_DIR)

    # 4. 评估量化模型
    acc_q, cm_q, _ = evaluate(quantized_model, device, test_loader, True)
    print(f"Quantized (Hardware) Acc: {acc_q * 100:.2f}%")

    # 5. 保存结果
    generate_report(OUTPUT_DIR, acc_fp, acc_q, args.dataset)

    # 绘制混淆矩阵对比
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_fp, annot=False, cmap='Greens', cbar=False)
    plt.title(f'Software (Acc: {acc_fp * 100:.1f}%)')
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_q, annot=False, cmap='Blues', cbar=False)
    plt.title(f'Hardware (Acc: {acc_q * 100:.1f}%)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_compare.png'))


if __name__ == '__main__':
    main()