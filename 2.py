# python
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


def get_data_loaders(dataset_name='mnist', batch_size=64):
    if dataset_name.lower() == 'fashion_mnist':
        print("Loading Fashion-MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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


# --- 3. 基于LUT的后量化模块 ---

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


# --- 4. 可视化与报告模块 ---

def generate_results(output_dir, history, dataset_name):
    epochs = range(1, len(history['train_losses']) + 1)
    dataset_title = dataset_name.replace('_', ' ').title()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_losses'], 'b-o', label='Training Loss')
    plt.title(f'Training Loss Curve ({dataset_title})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [acc * 100 for acc in history['test_accuracies']], 'r-o', label='Test Accuracy')
    plt.title(f'Test Accuracy Curve ({dataset_title}, Full Precision)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(history['computer_cm'], annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix ({dataset_title}, 64-bit Computer)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_computer.png'))
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(history['npnn_cm'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_title}, N-PNN with LUT)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_npnn.png'))
    plt.close()

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
    print(f"Loaded non-uniform hardware LUT with {len(final_hardware_lut)} levels.")
    print(f"LUT range: [{final_hardware_lut.min():.4f}, {final_hardware_lut.max():.4f}]")

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

    # 可选：保存权重与过程
    torch.save(model.state_dict(), os.path.join(output_dir, f'ocnn_model_fp32_{dataset_name}.pth'))
    with open(os.path.join(output_dir, f'ocnn_results_{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(history, f)

    generate_results(output_dir, history, dataset_name)

    # 汇总返回
    return {
        'dataset': dataset_name,
        'computer_accuracy': float(computer_accuracy),
        'npnn_accuracy': float(npnn_accuracy),
        'output_dir': output_dir
    }


# --- 6. 主执行流程：支持一次跑完两个数据集 ---

def main():
    parser = argparse.ArgumentParser(description='PTQ Replication for N-PNN on MNIST and Fashion-MNIST')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'mnist', 'fashion_mnist'],
                        help='Choose dataset: mnist | fashion_mnist | all (default: all)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per dataset')
    args = parser.parse_args()

    datasets_to_run = ['mnist', 'fashion_mnist'] if args.dataset == 'all' else [args.dataset]

    summaries = []
    for ds in datasets_to_run:
        print(f"\n================= Running pipeline for {ds.upper()} =================")
        summary = run_pipeline_for_dataset(ds, num_epochs=args.epochs)
        summaries.append(summary)

    # 合并摘要
    with open('results_summary_all.txt', 'w') as f:
        f.write('--- Summary over datasets ---\n')
        for s in summaries:
            f.write(f"{s['dataset']}: FP32={s['computer_accuracy'] * 100:.2f}%, "
                    f"Q(LUT)={s['npnn_accuracy'] * 100:.2f}%, "
                    f"dir='{s['output_dir']}'\n")
    print("\nSummary saved to 'results_summary_all.txt'")


if __name__ == '__main__':
    main()
