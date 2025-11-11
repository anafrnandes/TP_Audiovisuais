import pathlib

dataset_path = pathlib.Path("dataset_waste_container")

print(f"A analisar o dataset em: {dataset_path}\n")

contagens = {}
total_imagens = 0

if not dataset_path.is_dir():
    print(f"Erro: O caminho '{dataset_path}' não foi encontrado ou não é uma pasta.")
    print("Por favor, verifica a variável 'dataset_path' no script.")
else:
    # Agora só precisamos de verificar se é uma pasta
    for pasta_classe in dataset_path.iterdir():
        if pasta_classe.is_dir():
            num_imagens = len(list(pasta_classe.glob('*')))
            contagens[pasta_classe.name] = num_imagens
            total_imagens += num_imagens

    print("Contagem de Imagens por Classe (7 classes)")
    for nome_pasta, contagem in sorted(contagens.items()):
        print(f"- {nome_pasta}: {contagem} imagens")

    print(f"Total de imagens: {total_imagens}")