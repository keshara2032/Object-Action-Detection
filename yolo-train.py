import sys
import torch
import torchvision
from ultralytics import YOLO
from roboflow import Roboflow

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
print(torch.version.cuda)


# !nvidia-smi
if __name__ == '__main__':  


    # rf = Roboflow(api_key="8OdfbI9Sz6uvYfFcsBZh")
    # project = rf.workspace("cognitiveems-project").project("cognitiveems")
    # dataset = project.version(1).download("yolov8")


    model = YOLO('yolov8n.pt')

    
    # Training.
    results = model.train(
    data='./datasets/data.yaml',
    imgsz=640,
    epochs=200,
    batch=8,
    name='yolov8n_custom')