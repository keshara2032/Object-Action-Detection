import sys
import torch
import torchvision
from ultralytics import YOLO

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
print(torch.version.cuda)

# !nvidia-smi
if __name__ == '__main__':  
    model = YOLO('yolov8n.pt')

    
    # Training.
    results = model.train(
    data='./datasets/data.yaml',
    imgsz=640,
    epochs=10,
    batch=8,
    name='yolov8n_custom')