import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pathlib
import os

# Desativa logs menos importantes do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Iniciando a avaliação do modelo...")

# 1. Configuração
MODEL_FILE_PATH = "waste_classifier_model.keras"  # O modelo que o train.py guardou
DATA_DIR = pathlib.Path("data_splits")
VAL_DIR = DATA_DIR / "validation"

# Parâmetros (têm de ser os mesmos do treino)
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32

# 2. Carregar o Modelo
if not pathlib.Path(MODEL_FILE_PATH).exists():
    print(f"Erro: Ficheiro do modelo não encontrado em '{MODEL_FILE_PATH}'")
    print("Por favor, corre o 'train.py' primeiro.")
    exit()

print(f"A carregar modelo de '{MODEL_FILE_PATH}'...")
model = tf.keras.models.load_model(MODEL_FILE_PATH)
print("Modelo carregado com sucesso.")

#  3. Carregar o Dataset de Validação
# É CRUCIAL que 'shuffle=False' aqui, para as labels
# e as previsões ficarem na ordem correta.
print(f"A carregar dados de validação de '{VAL_DIR}'...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names
print(f"Classes: {class_names}")

# 4. Obter Previsões e Labels Verdadeiras

print("A fazer previsões no set de validação...")
# model.predict() dá-nos as probabilidades de cada classe
raw_predictions = model.predict(validation_dataset)
# np.argmax() diz-nos qual é a classe com a maior probabilidade (a previsão final)
y_pred = np.argmax(raw_predictions, axis=1)

# Agora vamos extrair as labels verdadeiras (y_true) do dataset
y_true = np.concatenate([y for x, y in validation_dataset], axis=0)

print("Previsões concluídas.")

# 5. Gerar Relatório de Classificação
# Isto vai mostrar Precisão, Recall e F1-Score para CADA classe
print("\n" + "=" * 50)
print("      Relatório de Classificação (Fase 4)")
print("=" * 50)

report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# 6. Gerar Matriz de Confusão Visual
# Isto é perfeito para a tua apresentação e relatório
print("\n" + "=" * 50)
print("      A gerar Matriz de Confusão...")
print("=" * 50)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Matriz de Confusão')
plt.ylabel('Classe Verdadeira (True Label)')
plt.xlabel('Classe Prevista (Predicted Label)')
plt.tight_layout()  # Para garantir que os nomes das classes cabem

# Salvar a imagem
plot_filename = 'confusion_matrix.png'
plt.savefig(plot_filename)

print(f"\nMatriz de confusão guardada em '{plot_filename}'")
print("Avaliação concluída!")