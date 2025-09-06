import os
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from byol import BYOL
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


resnet = models.resnet152()

BATCH_SIZE = 32
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 2
IMAGE_SIZE = 256
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

ds = ImagesDataset("coco_train_2017", IMAGE_SIZE)
train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

model = BYOL(
        resnet,
        image_size=IMAGE_SIZE,
        hidden_layer='avgpool',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model = torch.nn.DataParallel(model)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in train_loader:
        imgs = batch.to(DEVICE)

        loss = model(imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.update_moving_average()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")
torch.save(model.net.state_dict(), os.path.join("pet_biometric_challenge_2022", "backbone_byol_res152.pt"))