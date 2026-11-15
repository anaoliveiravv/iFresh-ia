from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

model = YOLO("runs/classify/runs/ifresh/weights/best.pt")

test_dir = Path("dataset/test")
extensoes = {".jpg", ".jpeg", ".png", ".webp"}

imagens = sorted(
    arquivo
    for pasta in test_dir.iterdir()
    if pasta.is_dir()
    for arquivo in pasta.iterdir()
    if arquivo.suffix.lower() in extensoes
)

if not imagens:
    raise RuntimeError("Nenhuma imagem encontrada em dataset/test.")

nomes = model.names
classes = [nomes[i] for i in sorted(nomes)]
indices = {nome: i for i, nome in nomes.items()}

y_true = []
y_pred = []

for resultado in model.predict(
    source=[str(img) for img in imagens],
    batch=8,
    imgsz=224,
    verbose=False
):
    classe_real = resultado.path

    nome_pasta = Path(classe_real).parent.name

    y_true.append(indices[nome_pasta])
    y_pred.append(int(resultado.probs.top1))

acuracia = accuracy_score(y_true, y_pred)

print(f"Acurácia no teste: {acuracia:.2%}")

print("\nRelatório de classificação:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)

matriz = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes)))
)

fig, ax = plt.subplots(figsize=(10, 8))

ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=classes
).plot(ax=ax, xticks_rotation=45, cmap="Blues")

plt.title("iFresh - Matriz de Confusão")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()