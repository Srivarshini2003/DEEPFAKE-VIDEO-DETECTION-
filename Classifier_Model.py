import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json  # Importing json module
import os

# 1. Classifier definition
class DeepfakeClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=2):
        super(DeepfakeClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# 2. Dataset for frames + pseudo labels
class FrameDataset(Dataset):
    def __init__(self, json_path, encoder, transform=None):
        # Load JSON data
        with open(json_path, 'r') as f:
            self.labels = json.load(f)

        self.encoder = encoder.eval()  # Freeze encoder
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Build a list of samples
        self.samples = []
        for video_id, label in self.labels.items():
            video_folder = os.path.join(DATA_PATH, video_id)  # Assuming DATA_PATH is defined
            if os.path.isdir(video_folder):
                for frame in os.listdir(video_folder):
                    if frame.endswith('.jpg'):
                        frame_path = os.path.join(video_folder, frame)
                        self.samples.append({'image_path': frame_path, 'label': label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Access the sample dictionary using 'image_path' and 'label' keys
        img_path = self.samples[idx]['image_path']
        label = int(self.samples[idx]['label'])

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dim, no .cuda()

        with torch.no_grad():
            feature = self.encoder(img_tensor).squeeze(0)  # Remove batch dim

        return feature, label

# ... (rest of the code remains unchanged)

# 3. Load encoder
# Assuming EncoderCNN is defined elsewhere in your code

import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, base_model=None):
        super(EncoderCNN, self).__init__()
        if base_model is None:
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final classification layer (fc)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])  # output: (batch, 512, 1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x.view(x.size(0), -1)  # flatten to (batch, 512)


encoder = EncoderCNN()  # This is your EncoderCNN class
encoder.load_state_dict(torch.load("/content/drive/MyDrive/dataset/encoder_contrastive1.pth", map_location=torch.device('cpu'))) # Load to CPU
encoder.eval()

# 4. Dataset and DataLoader
json_path = "/content/drive/MyDrive/dataset/pseudo_labels1.json"  # JSON with image paths and labels
dataset = FrameDataset(json_path, encoder)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Train classifier
model = DeepfakeClassifier() # No .cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for features, labels in loader:
        # features, labels = features.cuda(), labels.cuda()  # Removed .cuda() calls

        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

    # Save the trained classifier model
save_path = "/content/drive/MyDrive/dataset/frame_classifier1.pth"
torch.save(model.state_dict(), save_path)
print(f"Classifier model saved to: {save_path}")
