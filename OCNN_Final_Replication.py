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

# 关键：从您提供的文件中导入真实的、非均匀的硬件查找表
from LUT import lut as hardware_lut

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. 模型与数据模块 ---

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


def get_data_loaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


# --- 2. 训练与评估模块 ---

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
    if not compute_details:
        return acc, None, None
    cm = confusion_matrix(all_targets, all_preds)
    digit_accs = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    return acc, cm, digit_accs


# --- 3. 核心：基于LUT的后量化模块 ---

def quantize_weights_with_lut(tensor, lut):
    tensor_np = tensor.cpu().detach().numpy()
    diff = np.abs(tensor_np[..., None] - lut[None, :])
    indices = np.argmin(diff, axis=-1)
    quantized = lut[indices]
    return torch.from_numpy(quantized).to(tensor.device)


def apply_ptq_with_lut(model, lut):
    quantized_model = copy.deepcopy(model)
    print("Applying PTQ with provided non-uniform LUT to model parameters...")
    for name, param in quantized_model.named_parameters():
        if param.dim() > 0:
            param.data = quantize_weights_with_lut(param.data, lut)
    print("Quantization complete.")
    return quantized_model


# --- 4. 可视化与报告模块 (已完善) ---

def generate_results(output_dir, history):
    epochs = range(1, len(history['train_losses']) + 1)

    # 1. 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o', label='Training Loss')
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 2. 绘制测试准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 'r-o', label='Test Accuracy')
    plt.title('Test Accuracy Curve (Full Precision Model)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    # 3. 绘制计算机（量化前）的混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(history['computer_cm'], annot=True, fmt='d', cmap='Greens')
    plt.title('Confusion Matrix (64-bit Computer)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_computer.png'))
    plt.close()

    # 4. 绘制N-PNN（量化后）的混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(history['npnn_cm'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (N-PNN Simulation with LUT)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_npnn.png'))
    plt.close()

    # 5. 生成最终的文本报告
    with open(os.path.join(output_dir, 'final_recognition_report.txt'), 'w') as f:
        f.write("--- Final PTQ Replication Results using Non-Uniform LUT ---\n\n")
        f.write(f"Full Precision Model Accuracy (64-bit Computer): {history['computer_accuracy'] * 100:.2f}%\n")
        f.write(f"Quantized Model Accuracy (N-PNN Simulation): {history['npnn_accuracy'] * 100:.2f}%\n\n")
        f.write("Per-Digit Accuracy (Full Precision):\n")
        for digit, acc in enumerate(history['computer_digit_accs']):
            f.write(f"  - Digit {digit}: {acc * 100:.2f}%\n")
        f.write("\nPer-Digit Accuracy (Quantized N-PNN with real LUT):\n")
        for digit, acc in enumerate(history['npnn_digit_accs']):
            f.write(f"  - Digit {digit}: {acc * 100:.2f}%\n")

    print(f"\nAll visualization results and the final report saved to '{output_dir}' folder")


# --- 5. 主执行流程 ---

def main():
    OUTPUT_DIR = 'results_final_ptq_replication'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    # --- STAGE 1: 全精度模型训练 ---
    model = DNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {
        'train_losses': [],
        'test_accuracies': []
    }

    print(f"Current device: {device}")
    print("--- STAGE 1: Starting Full Precision Model Training ---")
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, device, test_loader)

        # 记录每个周期的历史数据
        history['train_losses'].append(train_loss)
        history['test_accuracies'].append(acc)

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {acc * 100:.2f}%')

    print("\n--- Full precision training complete. ---")
    computer_accuracy, computer_cm, computer_digit_accs = evaluate(model, device, test_loader, compute_details=True)
    print(f'Final Full Precision Accuracy (64-bit Computer): {computer_accuracy * 100:.2f}%')

    # --- STAGE 2: 基于真实的、非均匀LUT的权重映射 ---
    print("\n--- STAGE 2: Mapping weights to the provided Non-Uniform Hardware LUT ---")

    final_hardware_lut = hardware_lut.astype(np.float32)
    print(f"Loaded non-uniform hardware LUT with {len(final_hardware_lut)} levels.")
    print(f"LUT range: [{final_hardware_lut.min():.4f}, {final_hardware_lut.max():.4f}]")

    quantized_model = apply_ptq_with_lut(model, final_hardware_lut)

    npnn_accuracy, npnn_cm, npnn_digit_accs = evaluate(quantized_model, device, test_loader, compute_details=True)
    print(f'Final Quantized Accuracy (N-PNN Simulation with real LUT): {npnn_accuracy * 100:.2f}%')

    # --- 保存最终结果 ---
    history['computer_accuracy'] = computer_accuracy
    history['computer_cm'] = computer_cm
    history['computer_digit_accs'] = computer_digit_accs
    history['npnn_accuracy'] = npnn_accuracy
    history['npnn_cm'] = npnn_cm
    history['npnn_digit_accs'] = npnn_digit_accs

    generate_results(OUTPUT_DIR, history)


if __name__ == '__main__':
    main()