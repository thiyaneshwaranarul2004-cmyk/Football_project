import numpy as np
from ultralytics import YOLO
# Load YOLOv8 model trained only on Indian currency
model = YOLO(r'C:\Users\PRIYADARSHENE S\Downloads\yolov8n.pt')  # Ensure this is trained only for Indian rupees

# Define dataset YAML (Ensure data.yaml exists in your working directory)
DATA_YAML_PATH = r"C:\Users\PRIYADARSHENE S\Desktop\football\data.yaml"

# Train the model using the correct data.yaml file
model.train(
    data=DATA_YAML_PATH,   # Correctly refer to the YAML file, not a dictionary
    epochs=50,          # Number of training epochs
    imgsz=340,         # Image size for training
    device="cpu"        # Use "cuda" if you have a GPU
)