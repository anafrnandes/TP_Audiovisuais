import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics

# Imports locais
from model import get_model  # <--- Mudou aqui
from utils import FocalLoss, get_transforms

# Configurações Globais
DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"
IMG_SIZE = 224
BATCH_SIZE = 32  # Com ResNet congelada, gasta menos memória, podes aumentar batch
EPOCHS = 20  # Transfer Learning convergil MUITO mais rápido
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "C:/python/TP_Audiovisuais/Configuracoes_Redes/Best_ResNet50_FineTuned.pth"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Treino")
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return running_loss / total, 100. * correct / total


def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return val_loss / total, 100. * correct / total


def main():
    if not os.path.exists("C:/python/TP_Audiovisuais/Configuracoes_Redes"):
        os.makedirs("C:/python/TP_Audiovisuais/Configuracoes_Redes")

    # 1. Dados
    train_transforms, val_transforms = get_transforms(IMG_SIZE)
    try:
        full_ds = datasets.ImageFolder(DATA_DIR)
        classes = full_ds.classes
        print(f"Classes: {classes}")
    except:
        print(f"Erro: Verifica '{DATA_DIR}'")
        return

    indices = list(range(len(full_ds)))
    split = int(np.floor(0.2 * len(full_ds)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]

    train_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=train_transforms), train_idx)
    val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_transforms), val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Pesos & Modelo
    print("A calcular pesos...")
    train_targets = [full_ds.targets[i] for i in train_idx]
    counts = Counter(train_targets)
    class_counts = [counts.get(i, 0) for i in range(len(classes))]
    weights = [sum(class_counts) / (len(classes) * c + 1e-5) for c in class_counts]
    class_weights = torch.FloatTensor(weights).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # --- AQUI ESTA O SEGREDO DO FINE TUNING ---
    model = get_model(num_classes=len(classes)).to(DEVICE)

    # Otimizamos APENAS a "head" (camadas finais), pois o resto está congelado
    # Isto torna o treino muito rápido e seguro
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    # 3. Loop
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"Iniciando Fine-Tuning da ResNet50 por {EPOCHS} épocas...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        v_loss, v_acc = validate(model, val_loader, criterion)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)

        print(f"   Train Acc: {t_acc:.2f}% | Val Acc: {v_acc:.2f}%")

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" Melhor modelo salvo ({best_acc:.2f}%)")

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy (Fine-Tuning)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()