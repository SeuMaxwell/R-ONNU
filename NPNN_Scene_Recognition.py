"""
NPNN_Scene_Recognition.py
=========================
Replicates the three scene recognition tasks described in the Advanced Optical
Materials paper for the N-PNN (Nonvolatile Photonic Neural Network) system,
using the real measured Sb₂Se₃ hardware LUT from LUT.py.

Tasks:
  1. Speech sequence recognition  (Free Spoken Digit Dataset / FSDD)
  2. Fashion image recognition     (Fashion-MNIST)
  3. Handwritten digit recognition (MNIST)

Usage:
  python NPNN_Scene_Recognition.py --task all      # run all three tasks
  python NPNN_Scene_Recognition.py --task speech   # speech only
  python NPNN_Scene_Recognition.py --task fashion  # Fashion-MNIST only
  python NPNN_Scene_Recognition.py --task mnist    # MNIST only
"""

import argparse
import copy
import os
import random
import sys
import urllib.request
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# Real measured Sb₂Se₃ hardware LUT (DO NOT MODIFY LUT.py)
from LUT import lut as hardware_lut

# ── Named constants ───────────────────────────────────────────────────────────
# Tiny value added to denominators to prevent division by zero
EPSILON = 1e-9
# Noise amplitude relative to the class-characteristic sinusoidal signal used
# when generating synthetic speech-like data (≈ 5 % of full-scale amplitude)
NOISE_AMPLITUDE = 0.05
# Target total number of test samples across all 10 FSDD digit classes
FSDD_TEST_SAMPLES_TARGET = 300
# Maximum per-class test samples (caps the test set for very large FSDD datasets)
FSDD_MAX_PER_CLASS_TEST = 30

# ── Optional audio libraries (try in priority order) ─────────────────────────
try:
    import librosa
    _HAVE_LIBROSA = True
except ImportError:
    _HAVE_LIBROSA = False

try:
    import soundfile as sf
    _HAVE_SOUNDFILE = True
except ImportError:
    _HAVE_SOUNDFILE = False

try:
    import scipy.io.wavfile as wavfile
    _HAVE_SCIPY_WAV = True
except ImportError:
    _HAVE_SCIPY_WAV = False

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class ImageCNN2D(nn.Module):
    """
    2-D CNN for MNIST / Fashion-MNIST (28×28 single-channel images).
    Architecture from the paper:
      Conv2d(1,32,3,pad=1) → ReLU → MaxPool(2,2)
      Conv2d(32,64,3,pad=1) → ReLU → MaxPool(2,2)
      Linear(64*7*7,128) → ReLU
      Linear(128,10) → Softmax
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class SpeechCNN1D(nn.Module):
    """
    1-D CNN for FSDD speech classification.
    Architecture following the paper (first conv uses 4 kernels K₁-K₄):
      Conv1d(1, 4, kernel_size=K) → ReLU → AdaptiveMaxPool1d(50)
      Conv1d(4, 8, kernel_size=K) → ReLU → AdaptiveMaxPool1d(10)
      Linear(8*10, 128) → ReLU
      Linear(128, 10) → Softmax
    """

    def __init__(self, input_length=8000, kernel_size=80):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool1 = nn.AdaptiveMaxPool1d(50)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=5, padding=2)
        self.pool2 = nn.AdaptiveMaxPool1d(10)
        self.fc1 = nn.Linear(8 * 10, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_image_loaders(dataset_name='mnist', batch_size=64):
    """Return (train_loader, test_loader) for MNIST or Fashion-MNIST."""
    if dataset_name.lower() == 'fashion_mnist':
        print("Loading Fashion-MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_ds = datasets.FashionMNIST('./data', train=True,  download=True, transform=transform)
        test_ds  = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    else:
        print("Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=1000,       shuffle=False, num_workers=0)
    return train_loader, test_loader


# ── FSDD helpers ──────────────────────────────────────────────────────────────

class FSDDDataset(Dataset):
    """Loads pre-processed FSDD waveforms from a list of (waveform, label) pairs."""

    def __init__(self, samples):
        self.samples = samples  # list of (np.ndarray of shape (8000,), int)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        waveform, label = self.samples[idx]
        # shape: (1, 8000) – single-channel 1-D signal
        x = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return x, label


def _load_wav(path):
    """Load a WAV file using whichever library is available. Returns (data, sr)."""
    if _HAVE_LIBROSA:
        data, sr = librosa.load(path, sr=None, mono=True)
        return data.astype(np.float32), sr
    if _HAVE_SOUNDFILE:
        data, sr = sf.read(path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    if _HAVE_SCIPY_WAV:
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        # Normalise int16/int32 to [-1, 1]
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        return data, sr
    raise RuntimeError("No audio library available (librosa / soundfile / scipy).")


def _preprocess_waveform(data, sr, target_length=8000, target_sr=8000):
    """Resample (if needed), truncate / zero-pad to target_length, normalise."""
    # Simple nearest-neighbour resample when librosa is not available
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        indices = np.round(np.linspace(0, len(data) - 1, new_len)).astype(int)
        data = data[indices]

    if len(data) >= target_length:
        data = data[:target_length]
    else:
        data = np.pad(data, (0, target_length - len(data)))

    # Peak normalise
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak
    return data.astype(np.float32)


def _try_download_fsdd(data_root='./data/fsdd'):
    """
    Attempt to download FSDD from GitHub.  Returns the path to the
    'recordings' directory, or None on failure.
    """
    recordings_dir = os.path.join(data_root, 'recordings')
    if os.path.isdir(recordings_dir) and len(os.listdir(recordings_dir)) > 10:
        print(f"FSDD recordings found at {recordings_dir}")
        return recordings_dir

    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, 'fsdd.zip')
    url = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip'

    print(f"Downloading FSDD from {url} …")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None

    print("Extracting …")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_root)
    except Exception as exc:
        print(f"  Extraction failed: {exc}")
        return None

    # The ZIP extracts to 'free-spoken-digit-dataset-master/recordings/'
    extracted = os.path.join(data_root, 'free-spoken-digit-dataset-master', 'recordings')
    if os.path.isdir(extracted):
        return extracted

    print("  Could not locate recordings directory after extraction.")
    return None


def _generate_synthetic_fsdd(n_samples=3000, length=8000, seed=42):
    """
    Generate a synthetic 1-D classification dataset that mimics FSDD structure:
    10 classes, ``n_samples`` total, each sample a float32 array of ``length``.
    Each class has a characteristic frequency pattern so there is genuine signal.
    """
    rng = np.random.default_rng(seed)
    samples = []
    per_class = n_samples // 10
    t = np.linspace(0, 1.0, length, dtype=np.float32)
    freqs = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]  # Hz per class

    for label in range(10):
        for _ in range(per_class):
            freq = freqs[label] + rng.uniform(-20, 20)
            harmonic = rng.uniform(0.2, 0.5)
            wave = (np.sin(2 * np.pi * freq * t, dtype=np.float32)
                    + harmonic * np.sin(4 * np.pi * freq * t, dtype=np.float32)
                    + NOISE_AMPLITUDE * rng.standard_normal(length).astype(np.float32))
            peak = np.max(np.abs(wave))
            if peak > 1e-6:
                wave /= peak
            samples.append((wave, label))

    rng.shuffle(samples)
    return samples


def get_speech_loaders(batch_size=64, target_length=8000):
    """
    Return (train_loader, test_loader) for FSDD.
    Strategy:
      1. Try to load pre-existing recordings.
      2. Try to download from GitHub.
      3. Fall back to synthetic data (with a warning).
    """
    recordings_dir = _try_download_fsdd()

    if recordings_dir and os.path.isdir(recordings_dir):
        print(f"Loading FSDD recordings from {recordings_dir} …")
        samples = []
        wav_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
        for fname in sorted(wav_files):
            try:
                label = int(fname.split('_')[0])
            except ValueError:
                continue
            path = os.path.join(recordings_dir, fname)
            try:
                data, sr = _load_wav(path)
                wave = _preprocess_waveform(data, sr, target_length=target_length)
                samples.append((wave, label))
            except Exception as exc:
                print(f"  Skipping {fname}: {exc}")

        if len(samples) >= 100:
            print(f"  Loaded {len(samples)} FSDD samples.")
        else:
            print(f"  Too few samples ({len(samples)}), falling back to synthetic data.")
            recordings_dir = None

    if not recordings_dir:
        print("WARNING: Using SYNTHETIC speech-like data (FSDD unavailable).")
        samples = _generate_synthetic_fsdd(n_samples=3000, length=target_length)

    # Stratified 2700/300 split (27 per class for test, 270 for train)
    by_class = {i: [] for i in range(10)}
    for item in samples:
        by_class[item[1]].append(item)

    train_items, test_items = [], []
    for label in range(10):
        cls_items = by_class[label]
        random.shuffle(cls_items)
        n_total = len(cls_items)
        # Target ~10 % of the full dataset as test, capped at FSDD_MAX_PER_CLASS_TEST
        n_test  = max(1, n_total * FSDD_TEST_SAMPLES_TARGET // max(len(samples), 1))
        n_test  = min(n_test, n_total // 10, FSDD_MAX_PER_CLASS_TEST)
        test_items.extend(cls_items[:n_test])
        train_items.extend(cls_items[n_test:])

    print(f"  Speech split → Train: {len(train_items)}, Test: {len(test_items)}")

    train_ds = FSDDDataset(train_items)
    test_ds  = FSDDDataset(test_items)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=len(test_items),  shuffle=False, num_workers=0)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, device, loader, optimizer, criterion):
    """One training epoch. Targets are one-hot encoded for MSE loss."""
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, num_classes=10).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target_onehot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, device, loader, compute_details=False):
    """Return (accuracy, confusion_matrix_or_None, per_class_acc_or_None)."""
    model.eval()
    correct = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            if compute_details:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    acc = correct / len(loader.dataset)
    if not compute_details:
        return acc, None, None

    cm = confusion_matrix(all_targets, all_preds)
    per_class = cm.diagonal() / (cm.sum(axis=1) + EPSILON)
    return acc, cm, per_class


# ─────────────────────────────────────────────────────────────────────────────
# 4.  POST-TRAINING QUANTIZATION WITH HARDWARE LUT
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_lut(tensor_np, lut_np):
    """Map every value in tensor_np to the nearest entry in lut_np."""
    diff = np.abs(tensor_np[..., np.newaxis] - lut_np[np.newaxis, :])
    indices = np.argmin(diff, axis=-1)
    return lut_np[indices]


def apply_ptq_with_lut(model, lut):
    """
    Layer-wise dynamic range matching PTQ:
      1. w_max = max(|weights|)
      2. lut_max = max(|lut|)
      3. scale = w_max / lut_max
      4. w_norm = weights / scale
      5. Quantize w_norm to nearest LUT value
      6. weights = quantized_norm * scale   (restore amplitude)

    This simulates programming the trained weights onto the Sb₂Se₃ photonic chip.
    """
    q_model = copy.deepcopy(model)
    lut_np  = lut.astype(np.float64)
    lut_max = float(np.max(np.abs(lut_np)))

    print("Applying PTQ with hardware LUT …")
    for name, param in q_model.named_parameters():
        if param.dim() == 0:      # skip scalars
            continue
        w_np  = param.data.cpu().numpy().astype(np.float64)
        w_max = float(np.max(np.abs(w_np)))

        if w_max > 1e-9:
            scale     = w_max / lut_max
            w_norm    = w_np / scale
            w_q_norm  = _nearest_lut(w_norm, lut_np)
            w_q       = (w_q_norm * scale).astype(np.float32)
        else:
            w_q = w_np.astype(np.float32)
            scale = 0.0

        param.data = torch.from_numpy(w_q).to(param.device)
        print(f"  {name:30s} | w_max={w_max:.4f} | scale={scale:.4f}")

    print("Quantization complete.\n")
    return q_model


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATION AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def _class_labels(task_name):
    if task_name == 'fashion_mnist':
        return ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return [str(i) for i in range(10)]


def generate_results(output_dir, history, task_name):
    os.makedirs(output_dir, exist_ok=True)
    epochs      = range(1, len(history['train_losses']) + 1)
    title_name  = task_name.replace('_', ' ').title()
    class_names = _class_labels(task_name)

    # 1. Training loss curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history['train_losses'], 'b-o', ms=4, label='Training Loss (MSE)')
    ax.set_title(f'Training Loss Curve – {title_name}', fontsize=13)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    # 2. Test accuracy curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [a * 100 for a in history['test_accuracies']],
            'r-o', ms=4, label='Test Accuracy (full precision)')
    ax.set_title(f'Test Accuracy Curve – {title_name}', fontsize=13)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)'); ax.grid(True); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_curve.png'), dpi=150)
    plt.close(fig)

    # 3. Confusion matrix – full precision (64-bit computer)
    _plot_confusion_matrix(
        history['computer_cm'], class_names,
        f'Confusion Matrix – {title_name} (64-bit Computer)',
        os.path.join(output_dir, 'confusion_matrix_computer.png'),
        cmap='Greens',
    )

    # 4. Confusion matrix – N-PNN (after LUT quantization)
    _plot_confusion_matrix(
        history['npnn_cm'], class_names,
        f'Confusion Matrix – {title_name} (N-PNN with LUT)',
        os.path.join(output_dir, 'confusion_matrix_npnn.png'),
        cmap='Blues',
    )

    # 5. Per-class accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(class_names))
    w = 0.35
    ax.bar(x - w / 2, history['computer_digit_accs'] * 100, w,
           label='64-bit Computer', color='steelblue')
    ax.bar(x + w / 2, history['npnn_digit_accs'] * 100, w,
           label='N-PNN (LUT)', color='coral')
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0, 110)
    ax.set_title(f'Per-Class Accuracy Comparison – {title_name}', fontsize=13)
    ax.legend(); ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=150)
    plt.close(fig)

    # 6. Text report
    report_path = os.path.join(output_dir, 'final_recognition_report.txt')
    with open(report_path, 'w') as fh:
        fh.write(f"=== N-PNN Scene Recognition – {title_name} ===\n\n")
        fh.write(f"Full Precision  (64-bit Computer): {history['computer_accuracy'] * 100:.2f}%\n")
        fh.write(f"N-PNN Simulation (LUT Quantized):  {history['npnn_accuracy'] * 100:.2f}%\n")
        fh.write(f"Accuracy drop after PTQ:           "
                 f"{(history['computer_accuracy'] - history['npnn_accuracy']) * 100:.2f}%\n\n")
        fh.write("Per-Class Accuracy – Full Precision:\n")
        for i, cls in enumerate(class_names):
            fh.write(f"  Class {i:2d} ({cls:10s}): {history['computer_digit_accs'][i] * 100:.2f}%\n")
        fh.write("\nPer-Class Accuracy – N-PNN (LUT Quantized):\n")
        for i, cls in enumerate(class_names):
            fh.write(f"  Class {i:2d} ({cls:10s}): {history['npnn_digit_accs'][i] * 100:.2f}%\n")

    print(f"\n✓ Results saved to '{output_dir}/'")
    print(f"  Full precision accuracy : {history['computer_accuracy'] * 100:.2f}%")
    print(f"  N-PNN (LUT) accuracy    : {history['npnn_accuracy']      * 100:.2f}%")


def _plot_confusion_matrix(cm, class_names, title, save_path, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(10, 8))
    annot = len(class_names) <= 12   # annotate only for small matrices
    sns.heatmap(cm, annot=annot, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TASK RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_image_task(task_name, num_epochs, output_dir, batch_size=64):
    """Generic runner for MNIST and Fashion-MNIST."""
    print(f"\n{'='*60}")
    print(f" TASK: {task_name.upper().replace('_', '-')}")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = get_image_loaders(task_name, batch_size)

    model     = ImageCNN2D().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {'train_losses': [], 'test_accuracies': []}

    # ── Stage 1: full-precision training ────────────────────────────────────
    print(f"\n[Stage 1] Training full-precision model for {num_epochs} epochs …")
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, device, test_loader)
        history['train_losses'].append(loss)
        history['test_accuracies'].append(acc)
        print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}%")

    # Final full-precision evaluation
    computer_acc, computer_cm, computer_per_class = evaluate(model, device, test_loader,
                                                             compute_details=True)
    print(f"\n  → Full Precision Accuracy: {computer_acc * 100:.2f}%")

    # ── Stage 2: PTQ with real Sb₂Se₃ LUT ──────────────────────────────────
    print("\n[Stage 2] Applying PTQ with hardware LUT …")
    lut_fp32    = hardware_lut.astype(np.float32)
    q_model     = apply_ptq_with_lut(model, lut_fp32)
    npnn_acc, npnn_cm, npnn_per_class = evaluate(q_model, device, test_loader,
                                                  compute_details=True)
    print(f"  → N-PNN (LUT) Accuracy:    {npnn_acc * 100:.2f}%")

    # Save results
    history.update({
        'computer_accuracy':    computer_acc,
        'computer_cm':          computer_cm,
        'computer_digit_accs':  computer_per_class,
        'npnn_accuracy':        npnn_acc,
        'npnn_cm':              npnn_cm,
        'npnn_digit_accs':      npnn_per_class,
    })
    generate_results(output_dir, history, task_name)


def run_speech_task(num_epochs, output_dir, batch_size=64, target_length=8000):
    """Runner for FSDD speech recognition."""
    print(f"\n{'='*60}")
    print(f" TASK: SPEECH SEQUENCE RECOGNITION (FSDD)")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = get_speech_loaders(batch_size, target_length)

    model     = SpeechCNN1D(input_length=target_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    history = {'train_losses': [], 'test_accuracies': []}

    # ── Stage 1: full-precision training ────────────────────────────────────
    print(f"\n[Stage 1] Training full-precision model for {num_epochs} epochs …")
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, device, train_loader, optimizer, criterion)
        acc, _, _ = evaluate(model, device, test_loader)
        history['train_losses'].append(loss)
        history['test_accuracies'].append(acc)
        print(f"  Epoch {epoch:3d}/{num_epochs} | Loss: {loss:.4f} | Test Acc: {acc * 100:.2f}%")

    computer_acc, computer_cm, computer_per_class = evaluate(model, device, test_loader,
                                                             compute_details=True)
    print(f"\n  → Full Precision Accuracy: {computer_acc * 100:.2f}%")

    # ── Stage 2: PTQ with real Sb₂Se₃ LUT ──────────────────────────────────
    print("\n[Stage 2] Applying PTQ with hardware LUT …")
    lut_fp32    = hardware_lut.astype(np.float32)
    q_model     = apply_ptq_with_lut(model, lut_fp32)
    npnn_acc, npnn_cm, npnn_per_class = evaluate(q_model, device, test_loader,
                                                  compute_details=True)
    print(f"  → N-PNN (LUT) Accuracy:    {npnn_acc * 100:.2f}%")

    history.update({
        'computer_accuracy':    computer_acc,
        'computer_cm':          computer_cm,
        'computer_digit_accs':  computer_per_class,
        'npnn_accuracy':        npnn_acc,
        'npnn_cm':              npnn_cm,
        'npnn_digit_accs':      npnn_per_class,
    })
    generate_results(output_dir, history, 'speech')


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='N-PNN Scene Recognition – replicate paper results with hardware LUT'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['all', 'speech', 'fashion', 'mnist'],
        help=(
            'Which task(s) to run: '
            '"all" (default), "speech", "fashion" (Fashion-MNIST), "mnist"'
        ),
    )
    parser.add_argument('--epochs_mnist',   type=int, default=15,
                        help='Training epochs for MNIST (default: 15)')
    parser.add_argument('--epochs_fashion', type=int, default=20,
                        help='Training epochs for Fashion-MNIST (default: 20)')
    parser.add_argument('--epochs_speech',  type=int, default=40,
                        help='Training epochs for Speech/FSDD (default: 40)')
    parser.add_argument('--batch_size',     type=int, default=64,
                        help='Batch size for all tasks (default: 64)')
    args = parser.parse_args()

    # Print LUT info once at startup
    lut_np = hardware_lut.astype(np.float32)
    print(f"\nHardware LUT: {len(lut_np)} levels, "
          f"range [{lut_np.min():.4f}, {lut_np.max():.4f}]")

    run_mnist   = args.task in ('all', 'mnist')
    run_fashion = args.task in ('all', 'fashion')
    run_speech  = args.task in ('all', 'speech')

    if run_mnist:
        run_image_task(
            task_name='mnist',
            num_epochs=args.epochs_mnist,
            output_dir='results_mnist',
            batch_size=args.batch_size,
        )

    if run_fashion:
        run_image_task(
            task_name='fashion_mnist',
            num_epochs=args.epochs_fashion,
            output_dir='results_fashion_mnist',
            batch_size=args.batch_size,
        )

    if run_speech:
        run_speech_task(
            num_epochs=args.epochs_speech,
            output_dir='results_speech',
            batch_size=args.batch_size,
        )

    print("\n\n✓ All requested tasks completed.")


if __name__ == '__main__':
    main()
