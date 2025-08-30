
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn import init
import random
from itertools import combinations
from torch.optim import Adam, SGD
from torch.autograd import Variable

# %%
IMAGE_PATH = os.path.join('pet_biometric_challenge_2022', 'train', 'images')
DATA_PATH = os.path.join('pet_biometric_challenge_2022', 'train')
CSV_PATH = os.path.join(DATA_PATH, 'dog_pairs.csv')

# %%
df = pd.read_csv(CSV_PATH)

pairs = []

for _, row in df.iterrows():
    pairs.append((row['imageA'], row['imageB'], row['dogA'], row['dogB'], row['label']))
# %%
# train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, shuffle=True)
train_pairs = pairs

# %%
class SiameseDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, id1, id2, label = self.pairs[idx]
        
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

        return img1, img2, torch.tensor(id1, dtype=torch.int64), torch.tensor(id2, dtype=torch.int64), torch.tensor(label, dtype=torch.float32)

# %%
# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# %%
train_dataset = SiameseDataset(train_pairs, train_transform)
# val_dataset = SiameseDataset(val_pairs,val_transform)
print(f"Train size: {len(train_dataset)}")

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)


# %% [markdown]
# # BackBone (ResNet152)


class ResNet152_Backbone(nn.Module):
    def __init__(self):
        super(ResNet152_Backbone, self).__init__()

        resnet = models.resnet50()

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.extra_layers(x)
        return x

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


# %%
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=1024):
        """
        embedding_dim: final embedding dimension (paper uses 1024)
        """
        super().__init__()
        self.feature_extractor = ResNet152_Backbone()  # outputs (B,512,H,W)
        self.dam = DualAttentionModule(in_channels=512)  # outputs (B,1024,H,W)
        self.gap = nn.AdaptiveAvgPool2d((1,1))           

        self.fc = nn.Linear(3*512, embedding_dim, bias=False)  
        self.bn = nn.BatchNorm1d(embedding_dim)

    def embed(self, x):
        x = self.feature_extractor(x)   # (B,512,H,W)
        x = self.dam(x)                 # (B,1024,H,W)
        x = self.gap(x)                 # (B,1024,1,1)
        x = x.view(x.size(0), -1)       # (B,1024)
        x = self.fc(x)                  # (B, embedding_dim) (paper uses 1024)
        x = self.bn(x)

        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, img1, img2):
        emb1 = self.embed(img1)
        emb2 = self.embed(img2)
        return emb1, emb2

class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.5, easy_margin=False, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).
        
        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.num_classes = num_classes
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is modified from forward1, which takes
        52.49644996365532 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        """
        #print("Labels min:", ground_truth.min().item())
        #print("Labels max:", ground_truth.max().item())
        #print("Num classes:", self.num_classes)
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        # one_hot = torch.zeros(cos_theta.size(), device='cuda')
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        # output = cos_theta + one_hot
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss
# %%
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = label * torch.pow(euclidean_distance, 2)
        neg = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2) 
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive

# %%
def lr_lambda(epoch):
    if epoch < 100:
        return 1.0
    else:
        return max(0.0, 1.0 - (epoch - 100) / 100)


# %%


def train():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_model = SiameseNetwork().to(device)

    arcface = ArcFace(1024, 350).to(device)

    contrastive_loss = ContrastiveLoss().to(device)

    opt_model = Adam(siamese_model.parameters(),
                    lr=0.0001,
                    betas=(0.5, 0.999))

    opt_arcface = SGD(arcface.parameters(),
                    momentum=0.9,
                    weight_decay=0.0005)
    scheduler_1 = torch.optim.lr_scheduler.LambdaLR(opt_model, lr_lambda)
    scheduler_2 = torch.optim.lr_scheduler.LambdaLR(opt_arcface, lr_lambda)

    num_epochs = 200

    for epoch in range(num_epochs):
        for i, (img1, img2, id1, id2, labels) in enumerate(train_loader):
            img_set1 = Variable(img1.to(device))
            img_set2 = Variable(img2.to(device))
            id1 = Variable(id1.to(device))
            id2 = Variable(id2.to(device))
            labels = Variable(labels.view(-1, 1).float()).to(device)
            
            output1, output2 = siamese_model(img_set1, img_set2)

            loss = (arcface(output1, id1) + arcface(output2, id2)) / 2.0
            #loss_c = contrastive_loss(output1, output2, labels)
            #loss = loss_c + loss_a
            
            opt_arcface.zero_grad()
            opt_model.zero_grad()
            loss.backward()
            opt_arcface.step()
            opt_model.step()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        if (epoch + 1) % 50 == 0 and epoch < 199:
            torch.save(siamese_model.state_dict(), os.path.join("pet_biometric_challenge_2022", "200_dog_pairs", f"res50_arcface{epoch}.pt"))
        scheduler_1.step()
        scheduler_2.step()

    torch.save(siamese_model.state_dict(), os.path.join("pet_biometric_challenge_2022", "DNNet+res50Arcloss200.pt"))

train()



