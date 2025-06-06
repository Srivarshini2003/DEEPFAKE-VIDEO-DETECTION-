import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# === Encoder Definition ===
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)

# === Frame Classifier Definition ===
class FrameClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super(FrameClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.classifier(x)

# === Load Models ===
encoder = EncoderCNN()
encoder.load_state_dict(torch.load("/content/drive/MyDrive/dataset/encoder_contrastive1.pth", map_location='cpu'))
encoder.eval()

classifier = FrameClassifier()
classifier.load_state_dict(torch.load("/content/drive/MyDrive/dataset/frame_classifier1.pth", map_location='cpu'))
classifier.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Predict all frames in one video ===
def predict_video(video_path, threshold=0.5):
    frame_scores = []

    for frame_name in sorted(os.listdir(video_path)):
        if not frame_name.endswith('.jpg'):
            continue
        frame_path = os.path.join(video_path, frame_name)
        image = Image.open(frame_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            features = encoder(image_tensor)
            softmax_scores = classifier(features)
            fake_prob = softmax_scores[0][1].item()
            frame_scores.append(fake_prob)

    if not frame_scores:
        return 0, 0.0

    avg_score = np.mean(frame_scores)
    predicted_label = int(avg_score > threshold)
    return predicted_label, avg_score

# === Evaluate all video folders ===
def evaluate_videos(video_root, label_json, threshold=0.5):
    with open(label_json, 'r') as f:
        label_map = json.load(f)

    y_true = []
    y_pred = []

    for video_folder in os.listdir(video_root):
        video_path = os.path.join(video_root, video_folder)
        if not os.path.isdir(video_path):
            continue

        if video_folder not in label_map:
            print(f"⚠️ Skipping unlabeled video folder: {video_folder}")
            continue

        true_label = label_map[video_folder]
        pred_label, confidence = predict_video(video_path, threshold=threshold)

        y_true.append(true_label)
        y_pred.append(pred_label)

        print(f"\n📹 Video: {video_folder}")
        print(f"   ✅ Ground Truth: {'Fake' if true_label == 1 else 'Real'}")
        print(f"   🤖 Predicted   : {'Fake' if pred_label == 1 else 'Real'}")
        print(f"   🔍 Avg Confidence: {confidence:.4f}\n{'-'*50}")

    if not y_true:
        print("\n⚠️ No valid videos processed.")
        return

    print("\n📊 Video-Level Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Video-Level Confusion Matrix")
    plt.tight_layout()
    plt.show()

# === Set Dataset Path and Run ===
video_root = "/content/drive/MyDrive/testdata/processedip"
label_json = "/content/drive/MyDrive/testdata/pseudo_label.json"

evaluate_videos(video_root, label_json, threshold=0.5)
