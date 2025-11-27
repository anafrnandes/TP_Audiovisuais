import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

# Imports locais
from model import get_model
from utils import get_transforms

# Configurações
MODEL_PATH = "C:/python/TP_Audiovisuais/Configuracoes_Redes/Best_ResNet50_FineTuned.pth"
DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model():
    print(f"A carregar modelo de: {MODEL_PATH}")

    # 1. Preparar Dados (Tal como no treino)
    _, val_transforms = get_transforms(IMG_SIZE)

    try:
        full_ds = datasets.ImageFolder(DATA_DIR)
        classes = full_ds.classes
    except:
        print("Erro: Dataset não encontrado.")
        return

    # Recriar o split de validação (usando a mesma semente se possível,
    # ou assumindo o split aleatório para estatística geral)
    # Nota: O ideal era ter salvo os índices, mas para estatística geral isto serve
    indices = list(range(len(full_ds)))
    split = int(np.floor(0.2 * len(full_ds)))
    # Para ser consistente, usamos seed fixa aqui para tentar replicar
    np.random.seed(42)
    np.random.shuffle(indices)
    _, val_idx = indices[split:], indices[:split]

    val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_transforms), val_idx)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Carregar Modelo
    model = get_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("A calcular previsões...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 3. Métricas e Gráficos
    print("\n" + "=" * 30)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 30)
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Previsto')
    plt.title('Matriz de Confusão (Validação)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate_model()