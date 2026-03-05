import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
from sklearn.metrics import confusion_matrix

# 设置非交互式后端，避免Tcl/Tk依赖
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. QAT核心模块 ---

class FakeQuantizeSb2Se3(nn.Module):
    """
    为Sb2Se3器件定制的伪量化模块，用于量化感知训练。
    该模块模拟从全精度浮点权重到128个不对称物理级别的映射。
    """

    def __init__(self):
        super(FakeQuantizeSb2Se3, self).__init__()
        # 硬件的物理限制
        self.tau_min = -0.9995
        self.tau_max = 0.9274
        # 将物理范围映射到8-bit整数范围的低7-bit，即128个级别
        self.quant_min = 0
        self.quant_max = 127

    def forward(self, w_fp32):
        # (A) 动态计算缩放参数
        w_min, w_max = w_fp32.min(), w_fp32.max()

        # 防止 w_min == w_max 导致除零错误
        if torch.isclose(w_min, w_max):
            return w_fp32

        scale = (w_max - w_min) / (self.tau_max - self.tau_min)
        zero_point = self.quant_min - torch.round(self.tau_min / scale)

        # (B) 模拟量化与反量化
        w_quantized = torch.round(w_fp32 / scale + zero_point)
        w_quantized_clipped = torch.clamp(w_quantized, self.quant_min, self.quant_max)
        w_fake_quant = (w_quantized_clipped - zero_point) * scale

        # (C) 实现梯度直通 (Straight-Through Estimator, STE)
        return w_fp32 + (w_fake_quant - w_fp32).detach()


# --- 2. 模型与数据模块 ---

class DNN_QAT(nn.Module):
    def __init__(self):
        super(DNN_QAT, self).__init__()
        # 定义网络层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # 为每一层的权重和偏置创建一个量化器实例
        self.quant_w_conv1 = FakeQuantizeSb2Se3()
        self.quant_b_conv1 = FakeQuantizeSb2Se3()
        self.quant_w_conv2 = FakeQuantizeSb2Se3()
        self.quant_b_conv2 = FakeQuantizeSb2Se3()
        self.quant_w_fc1 = FakeQuantizeSb2Se3()
        self.quant_b_fc1 = FakeQuantizeSb2Se3()
        self.quant_w_fc2 = FakeQuantizeSb2Se3()
        self.quant_b_fc2 = FakeQuantizeSb2Se3()

    def forward(self, x):
        # 在使用权重和偏置前，先对其进行伪量化
        quant_weight_conv1 = self.quant_w_conv1(self.conv1.weight)
        quant_bias_conv1 = self.quant_b_conv1(self.conv1.bias)
        x = F.conv2d(x, quant_weight_conv1, quant_bias_conv1, padding=1)
        x = F.relu(x)
        x = self.pool1(x)

        quant_weight_conv2 = self.quant_w_conv2(self.conv2.weight)
        quant_bias_conv2 = self.quant_b_conv2(self.conv2.bias)
        x = F.conv2d(x, quant_weight_conv2, quant_bias_conv2, padding=1)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(-1, 64 * 7 * 7)

        quant_weight_fc1 = self.quant_w_fc1(self.fc1.weight)
        quant_bias_fc1 = self.quant_b_fc1(self.fc1.bias)
        x = F.linear(x, quant_weight_fc1, quant_bias_fc1)
        x = F.relu(x)

        quant_weight_fc2 = self.quant_w_fc2(self.fc2.weight)
        quant_bias_fc2 = self.quant_b_fc2(self.fc2.bias)
        x = F.linear(x, quant_weight_fc2, quant_bias_fc2)

        return F.softmax(x, dim=1)


def get_data_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


# --- 3. 训练与评估模块 ---

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
    all_preds = []
    all_targets = []

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
    if not compute_details:
        return acc, None, None

    cm = confusion_matrix(all_targets, all_preds)
    digit_accs = cm.diagonal() / cm.sum(axis=1)
    return acc, cm, digit_accs


# --- 4. 可视化与报告模块 ---

def generate_results(output_dir, history):
    epochs = range(1, len(history['train_losses']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o')
    plt.title('Training Loss Curve (QAT with MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve_qat.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 'r-o')
    plt.title('Test Accuracy Curve (QAT for N-PNN)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve_qat.png'))
    plt.close()

    # ... (Other plotting functions can be kept as they are) ...

    plt.figure(figsize=(10, 8))
    sns.heatmap(history['final_confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Final Confusion Matrix (QAT for N-PNN)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_qat.png'))
    plt.close()

    with open(os.path.join(output_dir, 'recognition_report_qat.txt'), 'w') as f:
        f.write("--- Quantization-Aware Training Results ---\n")
        f.write(f"Final Model Accuracy (Simulating N-PNN): {history['final_accuracy'] * 100:.2f}%\n\n")
        f.write("Per-Digit Final Accuracy:\n")
        final_digit_accs = history['digit_accuracies'][-1]
        for digit, acc in enumerate(final_digit_accs):
            f.write(f"  - Digit {digit}: {acc * 100:.2f}%\n")

    print(f"\nAll QAT visualization results saved to '{output_dir}' folder")


# --- 5. 主执行流程 ---

def main():
    OUTPUT_DIR = 'results_ocnn_qat'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    # Use the QAT-enabled model
    model = DNN_QAT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Current device: {device}")
    print("--- Starting Quantization-Aware Training (QAT) for N-PNN Simulation ---")

    history = {
        'train_losses': [], 'test_accuracies': [], 'digit_accuracies': []
    }
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, cm, digit_accs = evaluate(model, device, test_loader, compute_details=True)

        history['train_losses'].append(train_loss)
        history['test_accuracies'].append(acc)
        history['digit_accuracies'].append(digit_accs)

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy (Simulating N-PNN): {acc * 100:.2f}%')

    final_accuracy, final_cm, _ = evaluate(model, device, test_loader, compute_details=True)
    history['final_accuracy'] = final_accuracy
    history['final_confusion_matrix'] = final_cm
    print(f'\n--- Training Complete ---')
    print(f'Final QAT Model Accuracy (N-PNN Simulation): {final_accuracy * 100:.2f}%')

    # Save results and visualize
    generate_results(OUTPUT_DIR, history)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'ocnn_qat_model.pth'))
    with open(os.path.join(OUTPUT_DIR, 'ocnn_qat_results.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"QAT model and training results saved to '{OUTPUT_DIR}' folder.")


if __name__ == '__main__':
    main()