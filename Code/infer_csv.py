# 集成推理 + 完整评估 + 自动最优阈值搜索

import os
import cv2
import numpy as np
import pandas as pd
import torch
import timm
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

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
        elif i == 1: aug_img = cv2.flip(img, 1)
        elif i == 2: aug_img = cv2.flip(img, 0)
        elif i == 3: aug_img = cv2.flip(img, -1)
        elif i == 4: aug_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif i == 5: aug_img = cv2.rotate(img, cv2.ROTATE_180)
        elif i == 6: aug_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else: aug_img = img

        augmented = transform(image=aug_img)
        tensor = augmented['image'].unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor)).item()
        preds.append(prob)

    return np.mean(preds)

def load_model(ckpt_path, model_name, size, device):
    if 'b5' in model_name.lower():
        model_name = 'tf_efficientnet_b5'
    elif 'b7' in model_name.lower():
        model_name = 'tf_efficientnet_b7'

    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt if isinstance(ckpt, dict) and 'model' not in ckpt else ckpt.get('model', ckpt)
    if list(state_dict.keys())[0].startswith('module.'):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIGC Detection Inference + Full Evaluation")
    parser.add_argument('--csv', type=str, default='/home/stf/CourseWork/CS4487/Project/AIGC-Detection-Dataset-2025/dataset_csv/val.csv', help='包含 image_path,label,label_name 的CSV路径')
    parser.add_argument('--single_ckpt', type=str, default=None, help='单模型路径')
    parser.add_argument('--single_model', type=str, default=None, help='单模型名称')
    parser.add_argument('--single_size', type=int, default=384)

    parser.add_argument('--ensemble', type=str, nargs='+', default=None,
                        help='集成模型: ckpt:model_name:size:weight')

    parser.add_argument('--tta', type=int, default=8, help='TTA次数')
    parser.add_argument('--threshold', type=float, default=0.5, help='手动指定阈值，不指定则自动搜索最优')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} images from {args.csv}")

    # 加载模型
    models = []
    transforms = []
    weights = []

    if args.single_ckpt:
        print(f"Loading single model: {args.single_ckpt}")
        model, transform = load_model(args.single_ckpt, args.single_model, args.single_size, device)
        models = [model]
        transforms = [transform]
        weights = [1.0]
    elif args.ensemble:
        print(f"Loading ensemble of {len(args.ensemble)} models...")
        for item in args.ensemble:
            ckpt, name, size, w = item.split(':')
            size, w = int(size), float(w)
            print(f"  → {os.path.basename(ckpt)} | {name} | {size}x{size} | w={w}")
            model, transform = load_model(ckpt, name, size, device)
            models.append(model)
            transforms.append(transform)
            weights.append(w)
        weights = np.array(weights) / sum(weights)
    else:
        raise ValueError("必须指定 --single_ckpt 或 --ensemble")

    # 推理所有图片
    all_probs = []
    print("Starting inference with TTA...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"Warning: Not found: {img_path}")
            all_probs.append(0.5)
            continue

        probs = []
        for model, transform, weight in zip(models, transforms, weights):
            p = predict_tta(model, img_path, transform, device, tta=args.tta)
            probs.append(p * weight)
        final_prob = sum(probs)
        all_probs.append(final_prob)

    all_probs = np.array(all_probs)
    true_labels = df['label'].values.astype(int)

    # ==================== 自动最优阈值搜索 + 完整评估 ====================
    best_th = 0.5
    if args.threshold is None:
        print("\nSearching best threshold...")
        best_acc = 0
        for th in np.arange(0.05, 1.0, 0.001):
            pred = (all_probs > th).astype(int)
            acc = accuracy_score(true_labels, pred)
            if acc > best_acc:
                best_acc = acc
                best_th = th

        pred_labels = (all_probs > best_th).astype(int)
        args.threshold = best_th
    else:
        pred_labels = (all_probs > args.threshold).astype(int)


    # 计算所有指标
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, all_probs)

    # Youden J 最优阈值（参考）
    fpr, tpr, thr = roc_curve(true_labels, all_probs)
    youden_th = thr[np.argmax(tpr - fpr)]

    # ==================== 打印终极结果 ====================
    print("\n" + "="*60)
    print("               AIGC检测 终极评估报告")
    print("="*60)
    print(f"数据集数量       : {len(df)}")
    print(f"真实AI生成图     : {true_labels.sum()} (label=1)")
    print(f"真实真实图       : {len(true_labels) - true_labels.sum()} (label=0)")
    print(f"模型输出概率范围 : [{all_probs.min():.6f}, {all_probs.max():.6f}]")
    print(f"概率均值         : {all_probs.mean():.6f}")
    print("-"*60)
    print(f"【最优阈值】      : {best_th:.6f}")
    print(f"【当前使用阈值】  : {args.threshold:.6f}")
    print(f"【Youden J 阈值】  : {youden_th:.6f}")
    print("-"*60)
    print(f"Accuracy         : {acc*100:.6f}%")
    print(f"Precision        : {precision*100:.4f}%")
    print(f"Recall           : {recall*100:.4f}%")
    print(f"F1 Score         : {f1*100:.6f}%")
    print(f"AUC              : {auc*100:.6f}%")
    print("="*60)
    print("="*60)