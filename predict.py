import subprocess
from pathlib import Path
from ultralytics import YOLO

model = YOLO("runs/classify/runs/ifresh/weights/best.pt")

processo = subprocess.run(
    [
        "osascript",
        "-e",
        'POSIX path of (choose file with prompt "Selecione a foto do alimento")'
    ],
    capture_output=True,
    text=True
)

if processo.returncode != 0:
    print("Seleção cancelada.")
    raise SystemExit

imagem = Path(processo.stdout.strip())

resultado = model.predict(
    source=str(imagem),
    imgsz=224,
    verbose=False
)[0]

classe = resultado.names[resultado.probs.top1]
confianca = float(resultado.probs.top1conf) * 100

frutas = {
    "apples": "Maçã",
    "banana": "Banana",
    "oranges": "Laranja"
}

estragado = classe.startswith("rotten")
nome_fruta = classe.replace("rotten", "").replace("fresh", "")

fruta = frutas.get(nome_fruta, "Desconhecida")
estado = "DETERIORADO" if estragado else "FRESCO"

print("\n===== RESULTADO IFRESH =====")
print(f"Alimento: {fruta}")
print(f"Estado visual: {estado}")
print(f"Confiança: {confianca:.2f}%")