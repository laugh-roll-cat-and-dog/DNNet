
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


model_file = "byol_coco_res152_arc_network_b32.pt"
output_file = "pred_byol_coco_res152_DAM_arc_b32.csv"


IMAGE_PATH = os.path.join('pet_biometric_challenge_2022', 'train', 'images')
DATA_PATH = os.path.join('pet_biometric_challenge_2022', 'train')
CSV_PATH = os.path.join(DATA_PATH, 'validation_from_train.csv')


df = pd.read_csv(CSV_PATH)


val_pairs = [tuple(row) for row in df.values]


class SiameseDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        img1_full = os.path.join(IMAGE_PATH, img1_path)
        img2_full = os.path.join(IMAGE_PATH, img2_path)
        try:
            img1 = Image.open(img1_full).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image: {img1_full} — {e}")

        try:
            img2 = Image.open(img2_full).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image: {img2_full} — {e}")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32), img1_path, img2_path


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


val_dataset = SiameseDataset(val_pairs,val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)



class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()

    def forward(self, x):
        pam_out = self.pam(x)  
        cam_out = self.cam(x)  

        out = torch.cat([cam_out, pam_out, x], dim=1)
        return out

class Network(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        resnet = models.resnet152()
        resnet.load_state_dict((torch.load("pet_biometric_challenge_2022/byol_res152.pt", map_location=torch.device('cuda'))))
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.dam = DualAttentionModule(in_channels=512)  
        self.gap = nn.AdaptiveAvgPool2d((1,1))           

        self.fc = nn.Linear(3*512, embedding_dim, bias=False)  
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)   
        x = self.dam(x)                 
        x = self.gap(x)                 
        x = x.view(x.size(0), -1)      
        x = self.fc(x)             
        x = self.bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        return emb1, emb2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network().to(device)
model.load_state_dict(torch.load(f"pet_biometric_challenge_2022/{model_file}", map_location=device))
model.eval()


result = []


with torch.no_grad():
    for (img1, img2, labels, img1_path, img2_path) in val_loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        pred1, pred2 = model(img1, img2)
        similarity = F.cosine_similarity(pred1, pred2)
        print(f"Label: {labels}\nSimilarity: {similarity}\n------------------------------------------------")
        for i in range(len(img1_path)):
            result.append((img1_path[i], img2_path[i], labels[i], similarity[i]))


pred_df = pd.DataFrame(result, columns=['imageA', 'imageB', 'labels', 'similarity'])
for col in ['labels', 'similarity']:
    pred_df[col] = pred_df[col].astype(str) \
                    .str.replace('tensor\(', '', regex=True) \
                    .str.replace('\)', '', regex=True)

pred_df.to_csv(f"{output_file}")