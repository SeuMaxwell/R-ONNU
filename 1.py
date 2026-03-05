# -------------------------------------------
# 导入所需库
# -------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import copy
from sklearn.metrics import confusion_matrix
import matplotlib
import argparse

# 关键：从本地LUT.py文件导入真实的、非均匀的硬件查找表
from LUT2 import lut as hardware_lut

# 设置Matplotlib后端为'Agg'，使其可以在没有图形用户界面的服务器上运行并保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. 模型与数据模块 ---

class DNN(nn.Module):
    """
    定义用于MNIST和Fashion-MNIST识别的卷积神经网络（CNN）结构。
    结构包含：(Conv -> ReLU -> Pool) * 2 -> FC -> ReLU -> FC -> Softmax
    """

    def __init__(self):
        super(DNN, self).__init__()
        # 第一个卷积层：输入1个通道（灰度图），输出32个通道，卷积核大小3x3，填充为1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第一个池化层：2x2最大池化
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二个卷积层：输入32个通道，输出64个通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(2, 2)
        # 第一个全连接层：输入维度为 64 * 7 * 7 （经过两次2x2池化后28x28图像的尺寸），输出128个神经元
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 第二个全连接层：输出10个神经元，对应10个类别
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """定义模型的前向传播路径"""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # 将多维特征图展平为一维向量，以输入到全连接层
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        # 最终输出层使用Softmax激活函数，将输出转换为类别概率分布
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_data_loaders(dataset_name='mnist', batch_size=64):
    """
    根据给定的数据集名称，加载并预处理数据，返回训练和测试的数据加载器。
    :param dataset_name: 'mnist' 或 'fashion_mnist'
    :param batch_size: 批处理大小
    :return: train_loader, test_loader
    """
    if dataset_name.lower() == 'fashion_mnist':
        print("Loading Fashion-MNIST dataset...")
        # 为Fashion-MNIST定义数据转换，包括转换为Tensor和使用其特定的均值/标准差进行归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:  # 默认为MNIST
        print("Loading MNIST dataset...")
        # 为MNIST定义数据转换和归一化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建DataLoader，用于按批次加载数据
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


# --- 2. 训练与评估模块 ---

def train_epoch(model, device, train_loader, optimizer, criterion):
    """
    对模型进行一个完整的训练周期（epoch）。
    """
    model.train()  # 将模型设置为训练模式
    total_loss = 0
    # 遍历训练数据加载器中的所有批次
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # 将目标标签转换为one-hot编码，以适配MSE损失函数
        target_onehot = F.one_hot(target, num_classes=10).float()

        optimizer.zero_grad()  # 清除上一轮的梯度
        output = model(data)  # 前向传播，得到预测输出
        loss = criterion(output, target_onehot)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型权重
        total_loss += loss.item()
    return total_loss / len(train_loader)  # 返回平均损失


def evaluate(model, device, test_loader, compute_details=False):
    """
    在测试集上评估模型的性能。
    """
    model.eval()  # 将模型设置为评估模式
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():  # 在此代码块中，不计算梯度，以节省计算资源
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # 获取概率最高的类别的索引作为预测结果
            correct += pred.eq(target).sum().item()  # 累计预测正确的数量
            if compute_details:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    acc = correct / len(test_loader.dataset)  # 计算总体准确率
    if not compute_details:
        return acc, None, None

    # 如果需要，计算详细指标
    cm = confusion_matrix(all_targets, all_preds)  # 计算混淆矩阵
    digit_accs = cm.diagonal() / (cm.sum(axis=1) + 1e-9)  # 计算每个类别的准确率
    return acc, cm, digit_accs


# --- 3. 基于LUT的后量化模块 ---

def quantize_weights_with_lut(tensor, lut):
    """
    为张量中的每个值，在给定的查找表(LUT)中查找最近的邻居。
    这精确地模拟了论文中"The closest value is searched"的过程。
    """
    tensor_np = tensor.cpu().detach().numpy()
    # 使用NumPy的广播机制进行向量化计算，高效地找到每个元素的最近值索引
    diff = np.abs(tensor_np[..., None] - lut[None, :])
    indices = np.argmin(diff, axis=-1)
    quantized = lut[indices]  # 根据索引从LUT中获取量化后的值
    return torch.from_numpy(quantized).to(tensor.device)


def apply_ptq_with_lut(model, lut):
    """
    对整个模型应用基于LUT的训练后量化（PTQ）。
    """
    quantized_model = copy.deepcopy(model)  # 创建模型的深拷贝，以保留原始的全精度模型
    print("Applying PTQ with provided non-uniform LUT to model parameters...")
    # 遍历模型的所有参数（权重和偏置）
    for name, param in quantized_model.named_parameters():
        if param.dim() > 0:  # 确保参数不是标量
            param.data = quantize_weights_with_lut(param.data, lut)
    print("Quantization complete.")
    return quantized_model


# --- 4. 可视化与报告模块 ---

def generate_results(output_dir, history, dataset_name):
    """
    根据history字典中的数据，生成所有可视化图表和最终的文本报告。
    """
    epochs = range(1, len(history['train_losses']) + 1)
    dataset_title = dataset_name.replace('_', ' ').title()

    # 1. 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o', label='Training Loss')
    plt.title(f'Training Loss Curve ({dataset_title})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 2. 绘制测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 'r-o', label='Test Accuracy')
    plt.title(f'Test Accuracy Curve ({dataset_title}, Full Precision)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    # 3. 绘制计算机（量化前）的混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(history['computer_cm'], annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix ({dataset_title}, 64-bit Computer)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_computer.png'))
    plt.close()

    # 4. 绘制N-PNN（量化后）的混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(history['npnn_cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_title}, OCNN)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_ocnn.png'))
    plt.close()

    # 5. 生成最终的文本报告
    with open(os.path.join(output_dir, 'final_recognition_report.txt'), 'w') as f:
        f.write(f"--- Final PTQ Replication Results for {dataset_title} using Non-Uniform LUT ---\n\n")
        f.write(f"Full Precision Model Accuracy (64-bit Computer): {history['computer_accuracy'] * 100:.2f}%\n")
        f.write(f"Quantized Model Accuracy (N-PNN Simulation): {history['npnn_accuracy'] * 100:.2f}%\n\n")
        f.write("Per-Class Accuracy (Full Precision):\n")
        for i, acc in enumerate(history['computer_digit_accs']):
            f.write(f"  - Class {i}: {acc * 100:.2f}%\n")
        f.write("\nPer-Class Accuracy (Quantized N-PNN with real LUT):\n")
        for i, acc in enumerate(history['npnn_digit_accs']):
            f.write(f"  - Class {i}: {acc * 100:.2f}%\n")

    print(f"\nAll visualization results and the final report saved to '{output_dir}' folder")


# --- 5. 单数据集完整流水线 ---

def run_pipeline_for_dataset(dataset_name, num_epochs=10):
    """
    为单个数据集执行完整的“训练-量化-评估-报告”流水线。
    这是一个独立的、可复用的模块。
    """
    output_dir = f'results_{dataset_name}_final'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(dataset_name=dataset_name)

    # Stage 1: 全精度训练
    model = DNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {'train_losses': [], 'test_accuracies': []}

    print(f"Current device: {device}")
    print(f"--- STAGE 1: Starting Full Precision Model Training on {dataset_name.upper()} ---")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, device, test_loader)
        history['train_losses'].append(train_loss)
        history['test_accuracies'].append(acc)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {acc * 100:.2f}%')

    print("\n--- Full precision training complete. ---")
    computer_accuracy, computer_cm, computer_digit_accs = evaluate(model, device, test_loader, compute_details=True)
    print(f'Final Full Precision Accuracy (64-bit Computer): {computer_accuracy * 100:.2f}%')

    # Stage 2: LUT 映射与评估
    print("\n--- STAGE 2: Mapping weights to the provided Non-Uniform Hardware LUT ---")
    final_hardware_lut = hardware_lut.astype(np.float32)

    quantized_model = apply_ptq_with_lut(model, final_hardware_lut)
    npnn_accuracy, npnn_cm, npnn_digit_accs = evaluate(quantized_model, device, test_loader, compute_details=True)
    print(f'Final Quantized Accuracy (N-PNN Simulation with real LUT): {npnn_accuracy * 100:.2f}%')

    # 保存与可视化
    history['computer_accuracy'] = computer_accuracy
    history['computer_cm'] = computer_cm
    history['computer_digit_accs'] = computer_digit_accs
    history['npnn_accuracy'] = npnn_accuracy
    history['npnn_cm'] = npnn_cm
    history['npnn_digit_accs'] = npnn_digit_accs

    # 保存训练好的模型和历史数据，以备后续分析
    torch.save(model.state_dict(), os.path.join(output_dir, f'model_fp32_{dataset_name}.pth'))
    with open(os.path.join(output_dir, f'history_{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(history, f)

    generate_results(output_dir, history, dataset_name)

    # 返回一个摘要字典，用于最终的汇总报告
    return {
        'dataset': dataset_name,
        'computer_accuracy': float(computer_accuracy),
        'npnn_accuracy': float(npnn_accuracy),
        'output_dir': output_dir
    }


# --- 6. 主执行流程 ---

def main():
    """
    脚本的主入口。负责解析命令行参数，并调度执行一个或多个数据集的流水线。
    """
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='PTQ Replication for N-PNN on MNIST and Fashion-MNIST')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'mnist', 'fashion_mnist'],
                        help='Choose dataset: mnist | fashion_mnist | all (default: all)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per dataset (default: 10)')
    args = parser.parse_args()

    # 根据命令行参数决定要运行的数据集列表
    datasets_to_run = ['mnist', 'fashion_mnist'] if args.dataset == 'all' else [args.dataset]

    summaries = []
    # 循环调用流水线函数处理每个选定的数据集
    for ds in datasets_to_run:
        print(f"\n{'=' * 20} Running pipeline for {ds.upper()} {'=' * 20}")
        summary = run_pipeline_for_dataset(ds, num_epochs=args.epochs)
        summaries.append(summary)

    # 所有选择的数据集都运行完毕后，生成一个最终的汇总报告
    if len(summaries) > 1:
        with open('results_summary_all.txt', 'w') as f:
            f.write('--- Summary over all datasets ---\n')
            for s in summaries:
                f.write(f"Dataset: {s['dataset']}\n")
                f.write(f"  - FP32 Accuracy: {s['computer_accuracy'] * 100:.2f}%\n")
                f.write(f"  - Quantized (LUT) Accuracy: {s['npnn_accuracy'] * 100:.2f}%\n")
                f.write(f"  - Results saved in: '{s['output_dir']}'\n\n")
        print("\nOverall summary saved to 'results_summary_all.txt'")


# 确保只有在直接运行此脚本时，才调用main()函数
if __name__ == '__main__':
    main()