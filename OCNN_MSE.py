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

# 设置非交互式后端，避免Tcl/Tk依赖
import matplotlib

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
        # MODIFICATION: Added Softmax for MSE Loss
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

        # MODIFICATION: Convert target to one-hot encoding for MSE loss
        target_onehot = F.one_hot(target, num_classes=10).float()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_onehot)  # Use one-hot target
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

    # 计算详细指标
    cm = confusion_matrix(all_targets, all_preds)
    digit_accs = cm.diagonal() / cm.sum(axis=1)
    return acc, cm, digit_accs


# --- 3. 量化模块 ---

def quantize_weights(tensor, levels):
    tensor_np = tensor.cpu().detach().numpy()
    quantized = np.zeros_like(tensor_np)
    levels = np.array(levels)
    # Vectorized implementation for performance
    closest_indices = np.argmin(np.abs(tensor_np[..., None] - levels[None, :]), axis=-1)
    quantized = levels[closest_indices]
    return torch.from_numpy(quantized).to(tensor.device)


def apply_quantization(model, levels):
    quantized_model = copy.deepcopy(model)
    for name, param in quantized_model.named_parameters():
        if 'weight' in name:
            param.data = quantize_weights(param.data, levels)
    return quantized_model


# --- 4. 可视化与报告模块 ---

def generate_results(output_dir, history):
    epochs = range(1, len(history['train_losses']) + 1)

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o')
    plt.title('Training Loss Curve (OCNN with MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 'r-o')
    plt.title('Test Accuracy Curve (OCNN with MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    # 绘制各数字识别准确率
    plt.figure(figsize=(12, 8))
    digit_accuracies_T = np.array(history['digit_accuracies']).T
    for digit in range(10):
        plt.plot(epochs, digit_accuracies_T[digit] * 100, marker='o', label=f'Digit {digit}')
    plt.title('Digit Recognition Accuracy (OCNN with MSE)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'digit_accuracies.png'))
    plt.close()

    # 绘制最终混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(history['final_confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (OCNN with MSE)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # 生成报告
    with open(os.path.join(output_dir, 'recognition_report.txt'), 'w') as f:
        f.write("各数字最终识别准确率 (训练后):\n")
        final_digit_accs = history['digit_accuracies'][-1]
        for digit, acc in enumerate(final_digit_accs):
            f.write(f"数字 {digit}: {acc * 100:.2f}%\n")
        f.write("\n最容易混淆的数字对:\n")
        cm = history['final_confusion_matrix']
        # 查找非对角线上值最大的元素
        np.fill_diagonal(cm, 0)
        max_confused_val = np.max(cm)
        if max_confused_val > 0:
            indices = np.where(cm == max_confused_val)
            true_labels, pred_labels = indices[0], indices[1]
            for true_l, pred_l in zip(true_labels, pred_labels):
                f.write(f"  - 数字 {true_l} 最常被误识别为 {pred_l} ({max_confused_val} 次)\n")

    print(f"\n所有可视化结果已保存到 '{output_dir}' 文件夹")


# --- 5. 主执行流程 ---

def main():
    OUTPUT_DIR = 'results_ocnn_mse'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    model = DNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # MODIFICATION: Using MSELoss
    criterion = nn.MSELoss()

    print(f"当前使用的设备: {device}")
    print("--- 开始训练标准模型 (使用MSE Loss) ---")

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

        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {acc * 100:.2f}%')

    computer_accuracy, final_cm, _ = evaluate(model, device, test_loader, compute_details=True)
    history['final_confusion_matrix'] = final_cm
    print(f'\n最终计算机准确率: {computer_accuracy * 100:.2f}%')

    # 保存结果
    generate_results(OUTPUT_DIR, history)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'ocnn_model_before_quant.pth'))
    with open(os.path.join(OUTPUT_DIR, 'ocnn_results.pkl'), 'wb') as f:
        pickle.dump(history, f)
    print(f"模型和训练结果已保存到 '{OUTPUT_DIR}' 文件夹。")

    # --- 7-bit量化模拟N-PNN ---
    print("\n--- 开始模拟N-PNN量化 ---")
    levels = np.linspace(-0.9995, 0.9274, 128)
    quantized_model = apply_quantization(model, levels)
    print("模型权重已量化。")

    npnn_accuracy, _, _ = evaluate(quantized_model, device, test_loader)
    print(f'量化后 (N-PNN) 准确率: {npnn_accuracy * 100:.2f}%')

    # 将量化结果追加到报告中
    with open(os.path.join(OUTPUT_DIR, 'recognition_report.txt'), 'a') as f:
        f.write("\n\n--- 量化后性能 ---\n")
        f.write(f"计算机准确率: {computer_accuracy * 100:.2f}%\n")
        f.write(f"N-PNN (7-bit量化后) 准确率: {npnn_accuracy * 100:.2f}%\n")


if __name__ == '__main__':
    main()