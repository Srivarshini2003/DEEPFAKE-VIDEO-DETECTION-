# contrastive_learning/encoder.py

import os
import sys

# Get the current working directory
current_dir = os.getcwd()

# Add the directory containing your modules to sys.path
module_dir = os.path.join(current_dir, '/content/drive/MyDrive/Colab Notebooks/video-representation-learning') # Update with the actual path to your modules if different
sys.path.append(module_dir)

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# Now import your custom modules

# contrastive_learning/encoder.py

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

# contrastive_learning/projection_head.py

import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x):
        return self.mlp(x)



# Configuration
DATA_PATH = "/content/drive/MyDrive/dataset/preprocessd"
LABELS_JSON = "/content/drive/MyDrive/dataset/pseudo_labels1.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

# ... (Rest of the code remains the same)

# Two different augmentations for contrastive learning
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

class ContrastiveVideoFrameDataset(Dataset):
    def __init__(self, root_dir, label_json, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(label_json, 'r') as f:
            self.labels = json.load(f)
        self.samples = []
        for video_id, label in self.labels.items():
            video_folder = os.path.join(root_dir, video_id)
            if os.path.isdir(video_folder):
                for frame in os.listdir(video_folder):
                    if frame.endswith('.jpg'):
                        frame_path = os.path.join(video_folder, frame)
                        self.samples.append((frame_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, label = self.samples[idx]
        image = Image.open(frame_path).convert("RGB")
        if self.transform:
            image_i = self.transform(image)
            image_j = self.transform(image)
        return image_i, image_j

def contrastive_loss_fn(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim_matrix = sim_matrix / temperature

    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(DEVICE)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    positive_samples = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(DEVICE)
    loss = nn.functional.cross_entropy(sim_matrix, positive_samples)
    return loss

def train():
    dataset = ContrastiveVideoFrameDataset(DATA_PATH, LABELS_JSON, transform=augmentation)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    base_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder = EncoderCNN(base_encoder).to(DEVICE) # encoder is defined here
    projector = ProjectionHead(input_dim=512, projection_dim=128).to(DEVICE)

    model = nn.Sequential(encoder, projector)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images_i, images_j in dataloader:
            images_i = images_i.to(DEVICE)
            images_j = images_j.to(DEVICE)

            z_i = model(images_i)
            z_j = model(images_j)

            loss = contrastive_loss_fn(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

    return encoder # return the encoder instance


if __name__ == "__main__":
    encoder = train() # get the trained encoder from train()
    torch.save(encoder.state_dict(), "/content/drive/MyDrive/dataset/encoder_contrastive1.pth")
    print("Encoder saved successfully.")

