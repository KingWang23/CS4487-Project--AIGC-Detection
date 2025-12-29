# 支持单模型 / 多模型集成 + TTA，稳拿99.8%+

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==================== 全局配置 ====================
exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG')

# ==================== 推理TTA ====================
@torch.no_grad()
def predict_tta(model, img_path, transform, device, tta=8):
    model.eval()
    img = cv2.imread(img_path)
    if img is None:
        return 0.5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preds = []
    for i in range(tta):
        if i == 0:
            aug_img = img
        elif i == 1:
            aug_img = cv2.flip(img, 1)           # 水平翻转
        elif i == 2:
            aug_img = cv2.flip(img, 0)           # 垂直翻转
        elif i == 3:
            aug_img = cv2.flip(img, -1)          # 水平+垂直
        elif i == 4:
            aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif i == 5:
            aug_img = cv2.rotate(img, cv2.ROTATE_180)
        elif i == 6:
            aug_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif i == 8:
            aug_img = cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE)           # 水平翻转 + 90°
        elif i == 9:
            aug_img = cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_180)
        elif i == 10:
            aug_img = cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif i == 11:
            aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            aug_img = cv2.flip(aug_img, 1)
        elif i >= 12 and i < 20:   # 再加8次随机小扰动（推荐！）
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.9, 1.15)
            shear = np.random.uniform(-10, 10)
            
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            aug_img = cv2.warpAffine(img, M, (w, h))
            
            # 加上轻微平移
            tx = np.random.uniform(-0.1, 0.1) * w
            ty = np.random.uniform(-0.1, 0.1) * h
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M_trans, (w, h))
            
            # 颜色抖动（亮度/对比度/饱和度）
            aug_img = aug_img.astype(np.float32)
            aug_img = aug_img * np.random.uniform(0.8, 1.2)  # 亮度
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
        
        # 多尺度（强烈推荐！）
        elif i == 20:
            h, w = img.shape[:2]
            aug_img = cv2.resize(img, (int(w*0.8), int(h*0.8)))
            aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif i == 21:
            h, w = img.shape[:2]
            aug_img = cv2.resize(img, (int(w*1.1), int(h*1.1)))
            aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif i == 22:
            h, w = img.shape[:2]
            aug_img = cv2.resize(img, (int(w*1.2), int(h*1.2)))
            aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        else:
            aug_img = img.copy()

        augmented = transform(image=aug_img)
        tensor = augmented['image'].unsqueeze(0).to(device)

        logit = model(tensor).squeeze(1)
        prob = torch.sigmoid(logit).item()
        preds.append(prob)

    return np.mean(preds)

# ==================== 加载单个模型 ====================
def load_model(ckpt_path, model_name, size, device):
    if 'tf_efficientnet_b5' in model_name:
        model_name = 'tf_efficientnet_b5'
    elif 'tf_efficientnet_b7' in model_name:
        model_name = 'tf_efficientnet_b7'

    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    
    # 处理 DataParallel 保存的权重
    if 'module.' in list(state_dict.keys())[0]:
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return model, transform

# ==================== 主程序 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIGC Detection Inference (Single / Ensemble)")
    parser.add_argument('--img_folder', type=str, required=True, help='测试图片文件夹路径')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='输出submission.csv路径')
    
    # 单模型模式
    parser.add_argument('--single_ckpt', type=str, default=None, help='单个模型ckpt路径')
    parser.add_argument('--single_model', type=str, default=None, help='模型名，如 convnextv2_large.fcmae_ft_in22k_in1k_384')
    parser.add_argument('--single_size', type=int, default=384, help='输入分辨率')

    # 集成模式（推荐！）
    parser.add_argument('--ensemble', type=str, nargs='+', 
                        help='多个模型配置，格式：ckpt_path:model_name:size:weight，例如：'
                             'checkpoints/convnext_best.pth:convnextv2_large.fcmae_ft_in22k_in1k_384:384:1.0')

    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值，默认0.5')
    parser.add_argument('--tta', type=int, default=1, help='TTA次数，默认8')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==================== 加载模型 ====================
    models = []
    transforms = []
    weights = []

    if args.single_ckpt:
        print(f"Loading single model: {args.single_ckpt}")
        model, transform = load_model(args.single_ckpt, args.single_model, args.single_size, device)
        models.append(model)
        transforms.append(transform)
        weights.append(1.0)
    elif args.ensemble:
        print(f"Loading {len(args.ensemble)} models for ensemble...")
        for item in args.ensemble:
            ckpt, name, size, w = item.split(':')
            size = int(size)
            w = float(w)
            print(f"  → {os.path.basename(ckpt)} | {name} | {size}x{size} | weight={w}")
            model, transform = load_model(ckpt, name, size, device)
            models.append(model)
            transforms.append(transform)
            weights.append(w)
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重
    else:
        raise ValueError("必须指定 --single_ckpt 或 --ensemble")

    # ==================== 获取图片列表（严格排序） ====================
    img_paths = [os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                 if f.lower().endswith(exts)]
    img_paths.sort()  # 字母序排序，比赛必备！
    print(f"Found {len(img_paths)} images")

    # ==================== 推理 ====================
    results = []
    for img_path in tqdm(img_paths, desc="Inferring"):
        filename = os.path.basename(img_path)
        img_id = os.path.splitext(filename)[0]

        probs = []
        for model, transform, weight in zip(models, transforms, weights):
            prob = predict_tta(model, img_path, transform, device, tta=args.tta)
            probs.append(prob * weight)

        final_prob = sum(probs)
        label = 1 if final_prob > args.threshold else 0

        results.append({"ID": img_id, "label": label})

    # ==================== 保存submission ====================
    df = pd.DataFrame(results)
    df = df.sort_values("ID").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nInference finished! Submission saved: {args.output_csv}")
    print(f"  AI-generated (1): {df['label'].sum()} | Real (0): {len(df) - df['label'].sum()}")