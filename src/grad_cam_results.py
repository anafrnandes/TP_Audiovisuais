import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import random

# Imports locais
from model import get_model
from utils import get_transforms

# Configurações
MODEL_PATH = "C:/python/TP_Audiovisuais/Configuracoes_Redes/Best_ResNet50_FineTuned.pth"
DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"
SAVE_DIR = "gradcam_final_report"  # Nova pasta para não misturar
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_side_by_side():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. Carregar Modelo
    dummy_ds = datasets.ImageFolder(DATA_DIR)
    classes = dummy_ds.classes

    model = get_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Configurar Grad-CAM (ResNet Layer4)
    target_layer = model.layer4[-1]
    for param in target_layer.parameters():
        param.requires_grad = True
    cam = GradCAM(model=model, target_layers=[target_layer])

    _, val_transforms = get_transforms(IMG_SIZE)

    print(f"A gerar imagens 'Original vs Grad-CAM' para {len(classes)} classes...")

    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images: continue

        # Vamos tentar 3 vezes até encontrar uma que a previsão esteja correta (opcional, mas fica melhor no relatório)
        for _ in range(5):
            img_name = random.choice(images)
            img_path = os.path.join(class_path, img_name)

            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = val_transforms(img_pil).unsqueeze(0).to(DEVICE)

            # Previsão
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, predicted = probs.max(1)
            pred_class = classes[predicted.item()]

            # Se acertou (ou se quisermos mostrar erros também, remove este if)
            if pred_class == class_name:
                break

        # Gerar Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        img_np = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        # === CRIAR PLOT LADO A LADO ===
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Título Geral
        fig.suptitle(f"Real: {class_name} | Pred: {pred_class} ({conf.item():.1%})", fontsize=14, fontweight='bold')

        # Imagem 1: Original
        axes[0].imshow(img_np)
        axes[0].set_title("Imagem Original")
        axes[0].axis('off')

        # Imagem 2: Grad-CAM
        axes[1].imshow(visualization)
        axes[1].set_title("Ativação (Grad-CAM)")
        axes[1].axis('off')

        # Guardar
        save_path = os.path.join(SAVE_DIR, f"compare_{class_name}.jpg")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)  # dpi=150 para boa qualidade
        plt.close(fig)

        print(f"Gerado: {save_path}")


if __name__ == "__main__":
    generate_side_by_side()