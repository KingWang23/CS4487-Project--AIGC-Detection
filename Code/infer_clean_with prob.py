# 干净推理（无TTA） + 输出预测概率 + 支持单模型/多模型集成

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

# ==================== 与val完全一致的transform ====================
def get_val_transform(size=384):
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ==================== 单次前向（无任何TTA） ====================
@torch.no_grad()
def predict_single(model, img_path, transform, device):
    model.eval()
    img = cv2.imread(img_path)
    if img is None:
        return 0.5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    augmented = transform(image=img)
    tensor = augmented['image'].unsqueeze(0).to(device)

    logit = model(tensor).squeeze(1)
    prob = torch.sigmoid(logit).item()   # 0~1 之间的概率
    return prob

# ==================== 加载模型 ====================
def load_model(ckpt_path, model_name, device):
    if 'tf_efficientnet_b5' in model_name:
        model_name = 'tf_efficientnet_b5'
    elif 'tf_efficientnet_b7' in model_name:
        model_name = 'tf_efficientnet_b7'
    elif 'deit3' in model_name:
        model_name = 'deit3_large_patch16_384'

    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']

    if next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

# ==================== 主程序 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean Inference + Output Probability")
    parser.add_argument('--img_folder', type=str, required=True, help='测试图片文件夹')
    parser.add_argument('--output_csv', type=str, default='submission_with_prob.csv')

    # 单模型模式
    parser.add_argument('--ckpt', type=str, default=None, help='单模型ckpt路径')
    parser.add_argument('--model', type=str, default=None, help='模型名')
    parser.add_argument('--size', type=int, default=384, help='输入尺寸')

    # 多模型集成模式（推荐）
    parser.add_argument('--ensemble', type=str, nargs='+', default=None,
                        help='格式：ckpt:model_name:size:weight，例如 '
                             'ckpt1.pth:convnextv2_large.fcmae_ft_in22k_in1k_384:384:1.0')

    parser.add_argument('--threshold', type=float, default=0.5 ,
                        help='最终分类阈值（建议用验证集调到0.2~0.4）')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==================== 加载模型 ====================
    models_info = []

    if args.ensemble:
        print(f"Loading {len(args.ensemble)} models for ensemble...")
        for item in args.ensemble:
            ckpt, name, size, weight = item.split(':')
            size = int(size)
            weight = float(weight)
            model = load_model(ckpt, name, device)
            transform = get_val_transform(size)
            models_info.append({"model": model, "transform": transform, "weight": weight, "name": os.path.basename(ckpt)})
    elif args.ckpt:
        print(f"Loading single model: {args.ckpt}")
        model = load_model(args.ckpt, args.model, device)
        transform = get_val_transform(args.size)
        models_info.append({"model": model, "transform": transform, "weight": 1.0, "name": os.path.basename(args.ckpt)})
    else:
        raise ValueError("请指定 --ckpt 或 --ensemble")

    # 归一化权重
    total_w = sum(m["weight"] for m in models_info)
    for m in models_info:
        m["weight"] /= total_w

    # ==================== 推理 ====================
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG')
    img_paths = [os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                 if f.lower().endswith(exts)]
    img_paths.sort()
    print(f"Found {len(img_paths)} images")

    results = []
    for img_path in tqdm(img_paths, desc="Inferring (No TTA)"):
        filename = os.path.basename(img_path)
        img_id = os.path.splitext(filename)[0]

        # 集成预测概率
        final_prob = 0.0
        for info in models_info:
            prob = predict_single(info["model"], img_path, info["transform"], device)
            final_prob += prob * info["weight"]

        label = 1 if final_prob > args.threshold else 0
        results.append({
            "ID": img_id,
            "label": label,
            "prob": round(final_prob, 6)   # 保留6位小数
        })
    df = pd.DataFrame(results)
    df = df.sort_values("ID").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n推理完成！")
    print(f"Submission saved: {args.output_csv}")
    print(f"  AI-generated (1): {df['label'].sum()}")
    print(f"  Real (0): {len(df) - df['label'].sum()}")
    print(f"  Threshold used: {args.threshold}")
    print(f"  Prob range: {df['prob'].min():.6f} ~ {df['prob'].max():.6f}")
    print("\n概率分布统计：")
    print(df['prob'].describe())