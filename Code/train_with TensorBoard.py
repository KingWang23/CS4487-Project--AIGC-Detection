# train_pure_pytorch_with_tb.py
# 纯 PyTorch 训练脚本 + TensorBoard + F1监控 + 以验证F1保存最佳模型

import os
import random
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter      # <-- TensorBoard
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ==================== 1. 关键增强 ====================
import albumentations as A
from albumentations.pytorch import ToTensorV2

def dct_high_freq_boost(image, boost_prob=0.75, boost_factor=3.0, **kwargs):
    if random.random() > boost_prob:
        return image
    img = image.astype(np.float32)
    try:
        for i in range(3):
            dct = cv2.dct(cv2.dct(img[..., i], axis=0), axis=1)
            h, w = dct.shape
            mask_h, mask_w = h // 10, w // 10
            if mask_h > 0 and mask_w > 0:
                dct[mask_h:, mask_w:] *= np.random.uniform(1.3, boost_factor)
            img[..., i] = cv2.idct(cv2.idct(dct, axis=1), axis=0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    except:
        pass
    return img

def camera_artifacts(image, label):
    if label == 1:  # 真实图不加
        return image
    img = image.astype(np.float32)
    if random.random() < 0.75:
        q = random.randint(65, 94)
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(2, 5), img.shape)
        img += noise
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
    ])

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
            img = np.zeros((400, 400, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train and label == 0 and random.random() < 0.7:
            img = camera_artifacts(img, label)

        if self.transform:
            img = self.transform(image=img)['image']

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
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ==================== 4. 训练 & 验证（返回 Acc + F1）===
def train_one_epoch(model, loader, criterion_bce, criterion_focal, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    # 用于计算 Acc/F1

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss = criterion_bce(logits, labels) + 0.3 * criterion_focal(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            'loss': f'{total_loss/len(pbar):.4f}',
            'acc':  f'{accuracy_score(all_labels, all_preds):.5f}'
        })

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def validate(model, loader, criterion_bce, criterion_focal, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for imgs, labels in tqdm(loader, desc="Validating", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).squeeze(1)
        loss = criterion_bce(logits, labels) + 0.3 * criterion_focal(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc  = accuracy_score(all_labels, preds)
    f1   = f1_score(all_labels, preds)

    avg_loss = total_loss / len(loader)
    print(f"Val Loss: {avg_loss:.4f} | Acc@0.5: {acc:.6f} | F1@0.5: {f1:.6f} | AUC: {auc:.6f}")
    return avg_loss, acc, f1, auc


# ==================== 5. 主函数 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='convnextv2_large.fcmae_ft_in22k_in1k_384')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv',   type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs',     type=int, default=40)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--output',      type=str, default='checkpoints')
    parser.add_argument('--log_dir',    type=str, default='runs')   # TensorBoard 目录
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 自动分辨率
    size_dict = {
        'convnextv2_large.fcmae_ft_in22k_in1k_384': 384,
        'tf_efficientnet_b7.ns_jft_in1k': 384,
        'tf_efficientnet_b5.ns_jft_in1k': 512,
        'deit3_large_patch16_384.fb_in22k_ft_in1k': 384,
        'regnety_160': 384, 'regnety_320': 384,
        'swin_large_patch4_window12_384': 384,
        'swin_base_patch4_window12_384': 384,
    }
    size = size_dict.get(args.model, 384)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model: {args.model} | Input size: {size}")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.model}_bs{args.batch_size}_lr{args.lr}")
    print(f"TensorBoard: tensorboard --logdir={args.log_dir} --bind_all")

    # Dataset & Loader
    train_dataset = AIGCDataset(args.train_csv, get_train_aug(size), is_train=True)
    val_dataset   = AIGCDataset(args.val_csv,   get_val_aug(size),   is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size*2,
                              num_workers=8, pin_memory=True)

    model = timm.create_model(args.model, pretrained=True, num_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    criterion_bce   = nn.BCEWithLogitsLoss()
    criterion_focal = FocalLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 记录最佳指标
    best_acc = 0.0
    best_f1  = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion_bce, criterion_focal, optimizer, device, epoch, args.epochs)

        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion_bce, criterion_focal, device)

        scheduler.step()

        # TensorBoard 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Acc/train',  train_acc,  epoch)
        writer.add_scalar('Acc/val',    val_acc,    epoch)
        writer.add_scalar('F1/train',   train_f1,   epoch)
        writer.add_scalar('F1/val',     val_f1,     epoch)
        writer.add_scalar('AUC/val',    val_auc,    epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # ---------- 保存最佳 Acc 模型 ----------
        if val_acc > best_acc:
            best_acc   = val_acc
            best_f1    = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), f"{args.output}/{args.model}_best_acc.pth")
            print(f" → [NEW BEST] Epoch {epoch+1} | Val Acc@0.5 = {val_acc:.6f} ↑ | F1 = {val_f1:.6f}")


        print(f"Epoch {epoch+1:02d} | Train Loss:{train_loss:.4f} Acc:{train_acc:.5f} | "
              f"Val Loss:{val_loss:.4f} Acc:{val_acc:.6f} F1:{val_f1:.6f} AUC:{val_auc:.6f}\n")

    writer.close()
    print("="*70)
    print("Training finished!")
    print(f"Best Val Acc@0.5 : {best_acc:.6f} (at epoch {best_epoch+1})")
    print(f"Corresponding F1 : {best_f1:.6f}")
    print(f"Best model saved: {args.output}/{args.model}_best_acc.pth")
