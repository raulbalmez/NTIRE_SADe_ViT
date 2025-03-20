import onnxruntime as ort
import torch
import numpy as np
import cv2
import os
import argparse

# === Parse Command-Line Arguments ===
parser = argparse.ArgumentParser(description="Run ONNX model on images from a folder")
parser.add_argument("input_folder", type=str, help="Path to the input folder containing images")
parser.add_argument("output_folder", type=str, help="Path to the output folder for saving results")
args = parser.parse_args()

# === Assign Paths from Arguments ===
input_folder = args.input_folder
output_folder = args.output_folder
onnx_path = "model_restoration.onnx"  # Path to ONNX model

# === Ensure Output Folder Exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load ONNX Model ===
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === Process Each Image in Input Folder ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # === Read Image ===
        image = cv2.imread(input_path)  # Read image using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # === Normalize and Convert to Tensor ===
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
        image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # === Run ONNX Inference ===
        output = session.run([output_name], {input_name: image})
        output_image = output[0][0]  # Remove batch dimension

        # === Convert Output Back to Image ===
        output_image = np.transpose(output_image, (1, 2, 0))  # Convert to (H, W, C)
        output_image = (output_image * 255).clip(0, 255).astype(np.uint8)  # Denormalize

        # === Save the Enhanced Image ===
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        cv2.imwrite(output_path, output_image)
        print(f"✅ Processed: {filename} → {output_path}")

print("✅ All images processed successfully!")
