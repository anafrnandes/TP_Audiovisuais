import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

from model import get_model
from utils import get_transforms

MODEL_PATH = "C:/python/TP_Audiovisuais/Configuracoes_Redes/Best_ResNet50_FineTuned.pth"
DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"
IMG_PATH = "C:/python/TP_Audiovisuais/teste_fogo/amarelo.jpg"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image():
    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado! Corre o train.py primeiro.")
        return

    # Recuperar nomes das classes
    dummy_ds = datasets.ImageFolder(DATA_DIR)
    classes = dummy_ds.classes
    print(f"Classes: {classes}")

    # Carregar Modelo
    model = get_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # IMPORTANTE: Manter em eval() para o Dropout/BatchNorm funcionarem como teste
    model.eval()

    if not os.path.exists(IMG_PATH):
        print(f"Imagem {IMG_PATH} não encontrada.")
        return

    _, val_transforms = get_transforms(IMG_SIZE)
    img_pil = Image.open(IMG_PATH).convert('RGB')
    input_tensor = val_transforms(img_pil).unsqueeze(0).to(DEVICE)

    # Previsão normal
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = probs.max(1)
        pred_class = classes[predicted.item()]

    print(f"Previsão: {pred_class} | Confiança: {conf.item():.2%}")

    # === GRAD-CAM ===
    # Definir a camada alvo (último bloco da ResNet)
    target_layer = model.layer4[-1]

    # --- FIX PARA O ERRO 'NoneType' ---
    # Como a rede foi congelada no treino, temos de ativar os gradientes
    # nesta camada ESPECIFICAMENTE para o Grad-CAM funcionar.
    for param in target_layer.parameters():
        param.requires_grad = True
    # ----------------------------------

    target_layers = [target_layer]

    # Construir o GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # Gerar o mapa de calor
    # Nota: Não usamos torch.no_grad() aqui porque o GradCAM precisa deles!
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Original: {pred_class}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM\n(Onde a IA olhou)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    predict_image()