from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

dataset = Path("dataset")

train_dir = dataset / "train"
val_dir = dataset / "val"

if not train_dir.exists():
    raise FileNotFoundError("Pasta dataset/train não encontrada.")

for classe in train_dir.iterdir():
    if not classe.is_dir():
        continue

    imagens = [
        arquivo for arquivo in classe.iterdir()
        if arquivo.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    if len(imagens) < 2:
        continue

    treino, validacao = train_test_split(
        imagens,
        test_size=0.2,
        random_state=42
    )

    pasta_val = val_dir / classe.name
    pasta_val.mkdir(parents=True, exist_ok=True)

    for imagem in validacao:
        shutil.move(str(imagem), str(pasta_val / imagem.name))

    print(
        f"{classe.name}: "
        f"{len(treino)} treino | "
        f"{len(validacao)} validação"
    )

print("Dataset preparado com sucesso!")