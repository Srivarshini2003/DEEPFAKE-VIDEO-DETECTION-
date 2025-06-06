import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder definition
class EncoderCNN(nn.Module):
    def __init__(self, base_model=None):
        super(EncoderCNN, self).__init__()
        if base_model is None:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # (batch, 512, 1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # (batch, 512)

# Load encoder
encoder = EncoderCNN().to(DEVICE)
encoder.load_state_dict(torch.load("/content/drive/MyDrive/dataset/encoder_contrastive1.pth", map_location=DEVICE))
encoder.eval()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_vaf(video_folder):
    features = []
    for img_name in sorted(os.listdir(video_folder)):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(video_folder, img_name)
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feature = encoder(image_tensor)
            features.append(feature.cpu().numpy())

    if features:
        features = np.vstack(features)
        avg_feature = np.mean(features, axis=0)  # VAF: mean-pooled vector
        return avg_feature.tolist()
    else:
        return np.zeros(512).tolist()  # If no image found, return zeros

# Example usage: extract features for all video folders
input_root = "/content/drive/MyDrive/dataset/preprocessd"
output_json = "/content/drive/MyDrive/dataset/video_level_features.json"

import json

results = {}
print("[INFO] Extracting VAFs from video folders...")
for video_folder in tqdm(os.listdir(input_root)):
    full_path = os.path.join(input_root, video_folder)
    if os.path.isdir(full_path):
        vaf = extract_vaf(full_path)
        results[video_folder] = vaf

with open(output_json, 'w') as f:
    json.dump(results, f, indent=4)

print(f"[DONE] Extracted features for {len(results)} videos and saved to {output_json}")
