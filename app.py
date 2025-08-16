from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# 1️⃣ Load model architecture
model = models.resnext50_32x4d(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Load weights and move to device
model.load_state_dict(torch.load("saved_model.pth", map_location=device))
model = model.to(device)
model.eval()

app = FastAPI()

# 3️⃣ Image transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):

    # Open the image properly
    img = Image.open(file.file).convert("RGB")

    # Transform + add batch dimension + move to device
    img_tensor = test_transforms(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        predictions = model(img_tensor)
        probs = F.softmax(predictions, dim=1)
        index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][index].item() * 100

    return {"Status": labels[index], "Confidence Score": confidence}