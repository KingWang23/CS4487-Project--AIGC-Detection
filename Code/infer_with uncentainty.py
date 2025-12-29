# infer_ensemble.py
# 集成 + TTA + 自动输出最不确定/分歧最大的样本

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
from collections import defaultdict

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


# ==================== 加载模型 ====================
def load_model(ckpt_path, model_name, size, device):
    if 'tf_efficientnet_b5' in model_name:
        model_name = 'tf_efficientnet_b5'
    elif 'tf_efficientnet_b7' in model_name:
        model_name = 'tf_efficientnet_b7'

    model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_path_rate=0.0)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=True)
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
    parser = argparse.ArgumentParser(description="AIGC Detection Inference + 分歧分析神器")
    parser.add_argument('--img_folder', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    
    parser.add_argument('--single_ckpt', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--single_size', type=int, default=384)

    parser.add_argument('--ensemble', type=str, nargs='+', 
                        help='格式: ckpt_path:model_name:size:weight')

    parser.add_argument('--threshold', type=float, default=0.56, help='最终判别阈值')
    parser.add_argument('--tta', type=int, default=14)
    parser.add_argument('--topk', type=int, default=20, help='打印最不确定和分歧最大的TopK样本')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==================== 加载模型 ====================
    models = []
    transforms = []
    weights = []
    model_names = []

    if args.single_ckpt:
        model, transform = load_model(args.single_ckpt, args.single_model, args.single_size, device)
        models.append(model)
        transforms.append(transform)
        weights.append(1.0)
        model_names.append(os.path.basename(args.single_ckpt))
    elif args.ensemble:
        for item in args.ensemble:
            ckpt, name, size, w = item.split(':')
            size = int(size)
            w = float(w)
            print(f"Loading → {os.path.basename(ckpt)} | {name} | {size}x{size} | weight={w}")
            model, transform = load_model(ckpt, name, size, device)
            models.append(model)
            transforms.append(transform)
            weights.append(w)
            model_names.append(os.path.basename(ckpt))
        weights = np.array(weights)
        weights = weights / weights.sum()
    else:
        raise ValueError("必须指定 --single_ckpt 或 --ensemble")

    # ==================== 推理 + 记录详细信息 ====================
    img_paths = [os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                 if f.lower().endswith(exts)]
    img_paths.sort()
    print(f"Found {len(img_paths)} images")

    detailed_results = []  # 存所有样本的详细预测信息

    for img_path in tqdm(img_paths, desc="Inferring"):
        filename = os.path.basename(img_path)
        img_id = os.path.splitext(filename)[0]

        model_probs = []
        for model, transform, weight, mname in zip(models, transforms, weights, model_names):
            prob = predict_tta(model, img_path, transform, device, tta=args.tta)
            model_probs.append(prob)

        weighted_prob = sum(p * w for p, w in zip(model_probs, weights))
        label = 1 if weighted_prob > args.threshold else 0

        detailed_results.append({
            'ID': img_id,
            'filename': filename,
            'final_prob': weighted_prob,
            'label': label,
            'model_probs': model_probs.copy(),
            'model_names': model_names.copy(),
            'weights': weights.tolist()
        })

    # ==================== 生成提交文件 ====================
    df_submit = pd.DataFrame([{'ID': r['ID'], 'label': r['label']} for r in detailed_results])
    df_submit = df_submit.sort_values('ID').reset_index(drop=True)
    df_submit.to_csv(args.output_csv, index=False)

    # ==================== 分歧分析：最不确定的样本 ====================
    print(f"\n{'='*80}")
    print(f"Top {args.topk} 最不确定的样本（概率最接近阈值 {args.threshold}）")
    print(f"{'='*80}")
    
    uncertainty = [abs(r['final_prob'] - args.threshold) for r in detailed_results]
    top_uncertain_idx = np.argsort(uncertainty)[:args.topk]

    for idx in top_uncertain_idx:
        r = detailed_results[idx]
        print(f"{r['filename']:<35} → 最终概率: {r['final_prob']:.4f}  判为: {r['label']}  (极不确定！)")

    # ==================== 分歧分析：模型分歧最大的样本 ====================
    print(f"\n{'='*80}")
    print(f"Top {args.topk} 模型分歧最大的样本（标准差最大）")
    print(f"{'='*80}")

    variances = [np.var(r['model_probs']) for r in detailed_results]
    top_diverse_idx = np.argsort(variances)[-args.topk:][::-1]

    for idx in top_diverse_idx:
        r = detailed_results[idx]
        print(f"{r['filename']:<35} → 最终概率: {r['final_prob']:.4f} → 判为: {r['label']}")
        for name, prob in zip(r['model_names'], r['model_probs']):
            print(f"   ├─ {name:<30} → {prob:.4f}")
        print(f"   std: {np.std(r['model_probs']):.4f}  var: {variances[idx]:.6f}\n")

    # ==================== 最终统计 ====================
    n_ai = df_submit['label'].sum()
    n_real = len(df_submit) - n_ai
    print(f"推理完成！")
    print(f"提交文件: {args.output_csv}")
    print(f"AI生成(1): {n_ai}   真实照片(0): {n_real}   占比: {n_ai/len(df_submit)*100:.2f}%")
    print(f"建议：重点检查上面列出的“不确定”和“分歧最大”的样本，可能藏着漏网AI图或真实图被错杀！")