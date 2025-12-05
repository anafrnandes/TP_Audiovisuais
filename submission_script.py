import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch.nn.functional as F
from src.model import get_model
from src.utils import get_transforms

# Configurações
BASE_DIR = r'C:\python\TP_Audiovisuais\teste_fogo'
OUTPUT_CSV = 'resultados_ana.csv'  # Podes dar um nome mais profissional
MODEL_PATH = r"C:\python\TP_Audiovisuais\Configuracoes_Redes\Best_ResNet50_FineTuned.pth"

# IMPORTANTE: A ordem alfabética TEM de ser igual à das pastas de treino!
CLASSES = [
    'container_battery',
    'container_biodegradable',
    'container_blue',
    'container_default',
    'container_green',
    'container_oil',
    'container_yellow'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def setup_system():
    """
    Carrega o modelo e as transformações uma única vez no início.
    """
    print(f"A configurar sistema no dispositivo: {DEVICE}")

    # 1. Carregar Modelo
    model = get_model(num_classes=NUM_CLASSES)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Modelo carregado com sucesso!")
    else:
        print(f"ERRO: Modelo não encontrado em {MODEL_PATH}")
        exit()

    model.to(DEVICE)
    model.eval()

    # 2. Carregar Transformações (Apenas as de validação/teste!)
    _, val_transforms = get_transforms(img_size=224)

    return model, val_transforms


def classifier_score(image_path, model, transforms):
    """
    Recebe o caminho da imagem e o modelo carregado.
    Retorna as probabilidades (scores) para cada classe.
    """
    try:
        # Abrir e processar imagem
        img = Image.open(image_path).convert('RGB')
        input_tensor = transforms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            # Aplicar Softmax para ter probabilidades entre 0 e 1
            probs = F.softmax(outputs, dim=1)

        # Converter para array numpy simples
        return probs.cpu().numpy()[0]

    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        # Em caso de erro (imagem corrompida), retorna zeros
        return np.zeros(NUM_CLASSES)


def run_test():
    # 1. Preparar o sistema (Modelo e Transforms)
    model, transforms = setup_system()

    all_paths = []

    # Percorre recursivamente todas as subpastas
    print(f"A procurar imagens em: {BASE_DIR}")
    for root, _, files in os.walk(BASE_DIR):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_paths.append(os.path.join(root, filename))

    if not all_paths:
        print(f"Nenhuma imagem encontrada em '{BASE_DIR}'.")
        return

    print(f"Total de {len(all_paths)} imagens para classificar.")

    results_list = []

    # Loop principal
    for path in all_paths:
        # Chama a tua função de classificação
        scores = classifier_score(path, model, transforms)

        # Identificar a classe vencedora
        idx_pred = np.argmax(scores)
        confianca = scores[idx_pred]
        nome_classe = CLASSES[idx_pred]  # Converte 0 -> 'container_battery'

        nome_imagem = os.path.basename(path)  # Pega só no nome do ficheiro, não o caminho todo

        # Cria a linha para o CSV
        # Podes adaptar estas colunas conforme o professor preferir
        row = {
            'Imagem': nome_imagem,
            'Previsão': nome_classe,  # O nome da classe (ex: container_blue)
            'Confiança': f"{confianca:.2%}"  # Ex: 99.50%
        }

        results_list.append(row)

    # Cria o DataFrame e exporta
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print(f"\n Processo concluído!")
    print(f"Resultados exportados para: {OUTPUT_CSV}")
    print("Amostra dos resultados:")
    print(df_results.head())


if __name__ == '__main__':
    run_test()