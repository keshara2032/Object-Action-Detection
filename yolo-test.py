import sys
import torch
import torchvision
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("CUDA is available:", torch.cuda.is_available())
# print(torch.version.cuda)

# !nvidia-smi
if __name__ == '__main__':  
    model = YOLO('./runs/detect/yolov8n_custom/weights/best.pt')

    image = cv2.imread('./images/cardiac.jpg')


    # # Perform object detection on an image using the model
    # results = model(image)

    results = model.predict(source=image, save=True)  # save plotted images

    # print(results[0].boxes[3])
    # cv2.imshow('Yolov8',image)
    # plot_bboxes(image, results[0].boxes.boxes, score=False)

