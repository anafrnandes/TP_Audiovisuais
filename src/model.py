import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    # 1. Carregar a ResNet50 com pesos treinados na ImageNet
    # "DEFAULT" garante que trazemos os melhores pesos disponíveis
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # 2. Congelar os parâmetros da rede base (Feature Extractor)
    # Isto impede que o treino altere o conhecimento base da rede
    for param in model.parameters():
        param.requires_grad = False

    # 3. Substituir a "cabeça" da rede (Fully Connected Layer)
    # A ResNet original tem 2048 entradas na última camada linear
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512), # Estabiliza o treino da nova camada
        nn.ReLU(),
        nn.Dropout(0.4),     # Dropout para garantir que não decora
        nn.Linear(512, num_classes)
    )

    return model