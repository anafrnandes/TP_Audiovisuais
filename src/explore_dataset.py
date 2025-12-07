import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "C:/python/TP_Audiovisuais/dataset_waste_container"


def explorar():
    if not os.path.exists(DATA_DIR):
        print("Dataset não encontrado.")
        return

    classes = os.listdir(DATA_DIR)
    counts = {}

    print("Análise Exploratória do Dataset")
    for c in classes:
        path = os.path.join(DATA_DIR, c)
        if os.path.isdir(path):
            n_imgs = len(os.listdir(path))
            counts[c] = n_imgs
            print(f"Classe {c}: {n_imgs} imagens")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Distribuição de Classes no Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explorar()