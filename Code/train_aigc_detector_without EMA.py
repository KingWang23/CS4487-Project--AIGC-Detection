# train_pure_pytorch_no_ema.py
# 纯 PyTorch 版（无EMA版本），使用原始模型

import os
import random
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==================== 1. 关键增强 ====================
import albumentations as A
from albumentations.pytorch import ToTensorV2

def dct_high_freq_boost(image, boost_prob=0.75, boost_factor=3.0, **kwargs):
    """
    DCT高频增强 - 对真实图像增强高频分量
    """
    if random.random() > boost_prob:
        return image
    
    img = image.astype(np.float32)
    try:
        for i in range(3):
            dct = cv2.dct(cv2.dct(img[..., i], axis=0), axis=1)
            h, w = dct.shape
            mask_h, mask_w = h//10, w//10
            if mask_h > 0 and mask_w > 0:
                dct[mask_h:, mask_w:] *= np.random.uniform(1.3, boost_factor)
            img[..., i] = cv2.idct(cv2.idct(dct, axis=1), axis=0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    except:
        pass  # 万一图片太小导致 mask_h=0，直接跳过
    return img

def camera_artifacts(image, label):
    """
    对真实图像添加相机痕迹模拟
    """
    if label == 1:
        return image
    img = image.astype(np.float32)

    if random.random() < 0.75:
        q = random.randint(65, 94)
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)

    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(2, 5), img.shape)
        img += noise.astype(np.float32)

    if random.random() < 0.4:
        shift = random.randint(-3, 3)
        img[..., 1] = np.roll(img[..., 1], shift, axis=0)
        img[..., 2] = np.roll(img[..., 2], shift, axis=1)

    return np.clip(img, 0, 255).astype(np.uint8)

def get_train_aug(size=384):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        A.GaussNoise(p=0.4),
        A.OneOf([A.Blur(blur_limit=5), A.Sharpen()], p=0.3),
        A.Lambda(image=dct_high_freq_boost, p=0.75),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], p=1.0)

def get_val_aug(size=384):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==================== 2. Dataset ====================
class AIGCDataset(Dataset):
    def __init__(self, csv_file, transform=None, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        label = int(row['label'])

        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Cannot read {path}, using black image")
            img = np.zeros((400, 400, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 只对训练集 + 真实图加相机痕迹
        if self.is_train and label == 0:
            if random.random() < 0.7:
                img = camera_artifacts(img, label)

        # 数据增强
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        return img, torch.tensor(label, dtype=torch.float32)

# ==================== 3. Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()

# ==================== 4. 训练和验证函数 ====================
def train_one_epoch(model, loader, criterion_bce, criterion_focal, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} Train")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss_bce = criterion_bce(logits, labels)
        loss_focal = criterion_focal(logits, labels)
        loss = loss_bce + 0.3 * loss_focal

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f"{total_loss/len(pbar):.4f}", 
            'acc': f"{accuracy_score(all_labels, all_preds):.5f}"
        })

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def validate(model, loader, criterion_bce, criterion_focal, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    pbar = tqdm(loader, desc="Validating", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        loss = criterion_bce(logits, labels) + 0.3 * criterion_focal(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'val_loss': f'{total_loss/len(pbar):.4f}'})

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 计算 AUC 和最佳阈值
    auc = roc_auc_score(all_labels, all_probs)

    # 搜索最佳阈值
    best_acc = 0.0
    best_th = 0.5
    for th in np.arange(0.05, 0.95, 0.001):
        acc = accuracy_score(all_labels, (all_probs > th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    avg_loss = total_loss / len(loader)
    print(f"Val Loss: {avg_loss:.4f} | Acc: {best_acc:.6f} | AUC: {auc:.6f} | Best Threshold: {best_th:.4f}")
    
    return avg_loss, best_acc, auc, best_th, all_probs, all_labels

# ==================== 5. TTA 推理函数 ====================
@torch.no_grad()
def predict_tta(model, img_path, size=384, tta=8, device='cuda'):
    model.eval()
    img = cv2.imread(img_path)
    if img is None:
        return 0.5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_aug = get_val_aug(size)
    preds = []
    for i in range(tta):
        if i == 0:
            aug_img = img
        elif i < 4:
            aug_img = cv2.rotate(img, i-1)
        else:
            aug_img = cv2.flip(img if i == 4 else cv2.rotate(img, i-5), (i-4)%2)
        aug = base_aug(image=aug_img)
        tensor = aug['image'].unsqueeze(0).to(device)
        pred = torch.sigmoid(model(tensor)).item()
        preds.append(pred)
    return np.mean(preds)

# ==================== 6. 主函数 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnextv2_large.fcmae_ft_in22k_in1k_384',
                        choices=[
                            'convnextv2_large.fcmae_ft_in22k_in1k_384',
                            'tf_efficientnet_b7.ns_jft_in1k',
                            'tf_efficientnet_b5.ns_jft_in1k',
                            'deit3_large_patch16_384.fb_in22k_ft_in1k',
                            # 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
                            # # 'eva02_base_patch14_224.mim_in22k',
                            # ### NEW: RegNetY 系列 ###
                            # 'regnety_160',
                            # 'regnety_320',

                            # ### NEW: Swin 系列 ###
                            # 'swin_large_patch4_window12_384',
                            # 'swin_base_patch4_window12_384',
                        ])
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--output', type=str, default='checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 自动匹配分辨率
    size_dict = {
        'convnextv2_large.fcmae_ft_in22k_in1k_384': 384,
        'tf_efficientnet_b7.ns_jft_in1k': 384,
        'tf_efficientnet_b5.ns_jft_in1k': 512,
        'deit3_large_patch16_384.fb_in22k_ft_in1k': 384,
        # NEW: RegNetY 默认 resolution
        # 'regnety_160': 384,
        # 'regnety_320': 384,

        # # NEW: Swin 默认 resolution
        # 'swin_large_patch4_window12_384': 384,
        # 'swin_base_patch4_window12_384': 384,
    }
    size = size_dict[args.model]

    # 模型名称修正
    if 'tf_efficientnet_b5' in args.model:
        model_name = 'tf_efficientnet_b5'
    elif 'tf_efficientnet_b7' in args.model:
        model_name = 'tf_efficientnet_b7'
    else:
        model_name = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, Model: {model_name}, Size: {size}")

    # Dataset & Loader
    train_dataset = AIGCDataset(args.train_csv, transform=get_train_aug(size), is_train=True)
    val_dataset = AIGCDataset(args.val_csv, transform=get_val_aug(size), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            num_workers=8, pin_memory=True)

    # Model - 使用原始模型，无EMA
    model = timm.create_model(model_name, pretrained=True, num_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Loss & Optimizer & Scheduler
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_focal = FocalLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环 - 简化版，无EMA
    best_acc = 0.0
    best_threshold_final = 0.5

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_bce, criterion_focal, 
            optimizer, device, epoch, args.epochs
        )
        
        # 验证
        val_loss, val_acc, val_auc, best_th, val_probs, val_labels = validate(
            model, val_loader, criterion_bce, criterion_focal, device
        )

        # 学习率调度
        scheduler.step()

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.6f} | "
              f"Val AUC: {val_auc:.6f} | Best Threshold: {best_th:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_threshold_final = best_th

            # 直接保存原始模型，无EMA
            torch.save({
                'model': model.state_dict(),
                'threshold': best_th,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'epoch': epoch + 1,
            }, f"{args.output}/{model_name}_best.pth")
            print(f"  → New best! Acc: {val_acc:.6f} | Use threshold: {best_th:.4f}")

    print(f"\nTraining finished! Best val acc: {best_acc:.5f}")
    print(f"Best model saved at: {args.output}/{model_name}_best.pth")

    # 保存阈值信息
    import json
    final_info = {
        "best_val_acc": float(best_acc),
        "model_name": model_name,
        "input_size": size,
    }
