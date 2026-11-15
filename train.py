import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Dispositivo utilizado: {device}")

model = YOLO("yolo26n-cls.pt")

model.train(
    data="dataset",
    epochs=10,
    imgsz=224,
    batch=16,
    device=device,
    project="runs",
    name="ifresh"
)

print("Treinamento finalizado!")