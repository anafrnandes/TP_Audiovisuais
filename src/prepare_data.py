import os
import numpy as np
from torchvision import datasets
from torch.utils.data import Subset
from collections import Counter

DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"


def prepare_split():

    print("Preparação de Dados (Split Estratificado)")

    # Carregar dataset
    dataset = datasets.ImageFolder(DATA_DIR)
    total_imgs = len(dataset)
    targets = dataset.targets

    # Índices
    indices = list(range(total_imgs))
    split_idx = int(np.floor(0.2 * total_imgs))

    # Shuffle com seed para reprodutibilidade
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split_idx:], indices[:split_idx]

    print(f"Total Imagens: {total_imgs}")
    print(f"Treino (80%): {len(train_idx)}")
    print(f"Validação (20%): {len(val_idx)}")

    # Verificar estratificação (exemplo na classe rara)
    train_targets = [targets[i] for i in train_idx]
    val_targets = [targets[i] for i in val_idx]

    print("\nVerificação de Distribuição (Amostra):")
    print(f"Counts Treino: {Counter(train_targets)}")
    print(f"Counts Validação: {Counter(val_targets)}")

    return train_idx, val_idx


if __name__ == "__main__":
    prepare_split()