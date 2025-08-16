# 🐔 Chicken Disease Detection using Deep Learning

This project focuses on detecting **chicken diseases** from fecal images with an accuracy of **~94%**.  
The model is trained to classify chicken droppings into the following categories:

- **Coccidiosis**
- **Healthy**
- **New Castle Disease**
- **Salmonella**

---

## ✨ Features
- Built using **ResNeXt50 (32x4d)** architecture
- **Transfer Learning** applied for improved performance on small dataset
- Achieved **94% classification accuracy**
- REST API deployment with **FastAPI** + **Uvicorn**
- Accepts chicken fecal images and returns:
  - Predicted disease status
  - Confidence score

---

## 📊 Dataset
The dataset was prepared and cleaned using images from Kaggle.  
You can check the preprocessing & training notebook here:  
🔗 [Kaggle Dataset & Training Notebook](https://www.kaggle.com/code/ajheshbasnet/chiken-disease-detection)

---

## 🛠️ Tech Stack
- **Python**
- **PyTorch** (for training & inference)
- **Torchvision** (pretrained ResNeXt50 model)
- **FastAPI** (API framework)
- **Uvicorn** (ASGI server)
- **PIL** & **Torchvision Transforms** (image preprocessing)

---

## 🚀 Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chicken-disease-detection.git
   cd chicken-disease-detection
