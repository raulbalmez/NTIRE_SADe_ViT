# **SADe_ViT: Semantic-Aware Depthwise Separable Convolution Transformer for Low-Light Image Enhancement**  

SADe_ViT is a **Semantic-Aware Visual Transformer** designed for **low-light image enhancement**. It incorporates **Depthwise Separable Convolutions (DSC)** in the feed-forward network (FFN) to improve efficiency and performance. This model is specifically designed for the **NTIRE 2025** low-light enhancement challenge.  

---

## 🚀 Installation  

Before running the model, install the required dependencies:  

```bash
pip install onnxruntime onnxruntime-gpu torch numpy opencv-python argparse
```
---

## ▶️ Inference  

To produce the enhanced images, use the following command:

```bash
python inference.py path/to/input_folder path/to/output_folder
```
