import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.utils.class_weight import compute_class_weight

print(f"Versão do TensorFlow: {tf.__version__}")

# 1. Configuração dos Parâmetros

# Caminho para os dados já divididos (o que o 'prepare_data.py' criou)
data_dir = pathlib.Path("data_splits")
train_dir = data_dir / "train"
val_dir = data_dir / "validation"

# Parâmetros do modelo
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32  # Quantas imagens processar de cada vez
EPOCHS = 20      # Quantas vezes "ver" o dataset todo. Começamos com 20.
MODEL_FILE_NAME = "waste_classifier_model.keras" # Nome do ficheiro para guardar o modelo

# 2. Carregar os Dados

print("A carregar datasets de treino e validação...")
# O Keras lê as pastas 'train' e 'validation' e percebe as classes
# automaticamente a partir dos nomes das sub-pastas.
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode='int',  # As labels são 0, 1, 2... 6
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False  # Não é preciso baralhar a validação
)

# Guarda os nomes das classes (ex: 'container_blue', 'container_oil', ...)
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Classes encontradas ({num_classes}): {class_names}")


# 3. Calcular Class Weights (Para o Desequilíbrio)
# Esta é a nossa estratégia para forçar o modelo a dar
# mais importância às classes com poucas imagens.

print("A calcular 'class weights' para o desequilíbrio...")
# Temos de "extrair" todas as labels do dataset de treino
y_train = np.concatenate([y for x, y in train_dataset], axis=0)

# Usamos o scikit-learn para calcular os pesos
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Convertemos para um dicionário que o Keras percebe (ex: {0: 1.2, 1: 15.3, ...})
class_weights = dict(zip(np.unique(y_train), weights))
print("Pesos calculados:")
for i, name in enumerate(class_names):
    print(f"- {name} (ID: {i}): Peso = {class_weights[i]:.2f}")


# 4. Definir Camada de Data Augmentation
# Esta é a outra parte da nossa estratégia.
# Estas transformações só serão aplicadas aos dados de TREINO.
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2), # Gira até 20%
        layers.RandomZoom(0.2),   # Zoom até 20%
        layers.RandomContrast(0.2), # Muda o contraste
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ],
    name="data_augmentation",
)

# 5. Definir o Modelo (Transfer Learning)

print("A construir o modelo com Transfer Learning (MobileNetV2)...")

# 5.1. Carregar o Modelo Base (MobileNetV2)
# include_top=False significa que NÃO queremos a camada final
# da rede (que previa 1000 classes do ImageNet).
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False,
    weights='imagenet'
)

# 5.2. Congelar o Modelo Base
# Não queremos re-treinar a MobileNetV2 toda, só usá-la
# como um extrator de características.
base_model.trainable = False

# 5.3. Construir o Nosso Modelo Final
inputs = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# 1. Aplicar data augmentation (só acontece durante o treino)
x = data_augmentation(inputs)

# 2. Pré-processar a imagem para o formato que a MobileNetV2 espera
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# 3. Passar pela rede base (congelada)
x = base_model(x, training=False) # 'training=False' é importante

# 4. Juntar os resultados num único vetor
x = layers.GlobalAveragePooling2D()(x)

# 5. Adicionar uma camada de 'Dropout' para evitar overfitting
x = layers.Dropout(0.3)(x)

# 6. A NOSSA camada final: 7 neurónios (1 por classe)
# 'softmax' garante que as saídas somam 1 (são probabilidades)
outputs = layers.Dense(num_classes, activation='softmax')(x)

# Juntar tudo no modelo final
model = keras.Model(inputs, outputs)

# 6. Compilar o Modelo

print("A compilar o modelo...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy', # Usar esta porque as labels são inteiros
    metrics=['accuracy']
)

# Mostra um resumo da arquitetura do modelo
model.summary()


# 7. Treinar o Modelo

print(f"\nIniciando o treino por {EPOCHS} epochs...")

# O 'fit' é onde a magia acontece.
# Passamos os dados de treino, os de validação
# E o mais importante: os nossos 'class_weights'
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    class_weight=class_weights
)

print("\nTreino concluído!")

# 8. Salvar o Modelo

print(f"A guardar o modelo treinado em '{MODEL_FILE_NAME}'...")
model.save(MODEL_FILE_NAME)
print("Modelo guardado com sucesso.")


# 9. Plotar Gráficos de Treino (Opcional, mas MUITO útil)
# Isto ajuda-te a ver se o modelo está a aprender
# ou só a 'decorar' (overfitting).

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.savefig('training_plots.png') # Salva a imagem na pasta
print("Gráficos de treino guardados em 'training_plots.png'")
plt.show() # Se quiseres que o gráfico abra numa janela