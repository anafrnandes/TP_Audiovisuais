import pathlib
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

print("Iniciando a preparação dos dados...")

# 1. Configuração

# Caminho para o teu dataset original (o que explorámos antes)
original_dataset_path = pathlib.Path("dataset_waste_container")

# Caminho para a nova pasta onde vamos guardar os splits
# Esta pasta NÃO deve estar dentro do 'original_dataset_path'
# Coloca-a na raiz do teu projeto.
output_path = pathlib.Path("data_splits")

# Proporção dos dados para validação (20%)
val_split_size = 0.20

# Seed para garantir que o split é sempre o mesmo (reprodutibilidade)
random_seed = 42

# 2. Ler todos os ficheiros e labels

all_filepaths = []
all_labels = []

if not original_dataset_path.is_dir():
    print(f"Erro: O caminho original '{original_dataset_path}' não foi encontrado.")
else:
    for class_dir in original_dataset_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            for filepath in class_dir.glob('*'):
                all_filepaths.append(filepath)
                all_labels.append(class_name)

    print(f"Total de imagens encontradas: {len(all_filepaths)}")

    # 3. Fazer o Split Estratificado

    print("A fazer o split estratificado (80% treino, 20% validação)...")

    # Usamos o 'stratify=all_labels' para garantir que a proporção das classes
    # (especialmente as pequenas) se mantém nos dois conjuntos.
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_filepaths,
        all_labels,
        test_size=val_split_size,
        stratify=all_labels,
        random_state=random_seed
    )

    print(f"Total de imagens de treino: {len(train_paths)}")
    print(f"Total de imagens de validação: {len(val_paths)}")

    # 4. Criar a nova estrutura de pastas e copiar os ficheiros

    print("A criar nova estrutura de pastas e a copiar ficheiros...")


    # Função auxiliar para copiar os ficheiros
    def copy_files_to_split(filepaths, labels, split_name):
        for filepath, label in zip(filepaths, labels):
            # Define o caminho de destino (ex: 'data_splits/train/container_blue')
            dest_dir = output_path / split_name / label

            # Cria a pasta de destino se não existir
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copia o ficheiro
            shutil.copy(filepath, dest_dir)


    # Copiar ficheiros de treino
    copy_files_to_split(train_paths, train_labels, "train")

    # Copiar ficheiros de validação
    copy_files_to_split(val_paths, val_labels, "validation")

    # 5. Mostrar Resumo da Operação

    print("\nResumo da Divisão")
    print("\nContagem de Treino:")
    print(Counter(sorted(train_labels)))

    print("\nContagem de Validação:")
    print(Counter(sorted(val_labels)))

    print(f"\nFeito! O teu dataset pronto para o treino está em '{output_path}'")