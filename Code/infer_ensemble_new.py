#单模型 + 集成 + TTA + 验证集评估）

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ==================== 全局配置 ====================
exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG')

# ==================== Dataset ====================
class AIGCDataset(Dataset):
    def __init__(self, csv_file=None, img_folder=None):
        self.samples = []

        if csv_file is not None:
            # 评估模式
            df = pd.read_csv(csv_file)
            assert {'image_path', 'label'}.issubset(df.columns), "csv 必须包含 image_path 和 label 列"
            for _, row in df.iterrows():
                self.samples.append({
                    'path': row['image_path'],
                    'label': int(row['label']),
                    'is_eval': True
                })
        else:
            # 提交模式
            assert img_folder is not None
            img_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)
                         if f.lower().endswith(exts)]
            img_paths.sort()
            for p in img_paths:
                img_id = os.path.splitext(os.path.basename(p))[0]
                self.samples.append({
                    'path': p,
                    'label': img_id,   # 占位，实际是 ID
                    'is_eval': False
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.imread(item['path'])
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {item['path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 不在这里做 transform，由主循环统一处理（支持多模型不同 size）
        return img, item['label'], item['path'], item['is_eval']


# ==================== 自定义 collate_fn ====================
def custom_collate_fn(batch):
    imgs, labels, paths, is_eval_flags = zip(*batch)
    # imgs 是 numpy 数组列表
    return list(imgs), list(labels), list(paths), list(is_eval_flags)


# ==================== TTA 批量推理 ====================
@torch.no_grad()
def predict_tta_batch(models, transforms, weights, batch_imgs_np, device, tta=8):
    batch_size = len(batch_imgs_np)
    batch_probs = np.zeros(batch_size, dtype=np.float32)

    for model, transform, weight in zip(models, transforms, weights):
        model.eval()
        probs_list = []

        for t in range(max(1, tta)):
            if t == 0:
                imgs = batch_imgs_np.copy()
            elif t == 1:
                imgs = [cv2.flip(img, 1) for img in batch_imgs_np]
            elif t == 2:
                imgs = [cv2.flip(img, 0) for img in batch_imgs_np]
            elif t == 3:
                imgs = [cv2.flip(img, -1) for img in batch_imgs_np]
            elif t == 4:
                imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in batch_imgs_np]
            elif t == 5:
                imgs = [cv2.rotate(img, cv2.ROTATE_180) for img in batch_imgs_np]
            elif t == 6:
                imgs = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in batch_imgs_np]
            else:
                # 随机轻微扰动
                imgs_aug = []
                for img in batch_imgs_np:
                    angle = np.random.uniform(-10, 10)
                    scale = np.random.uniform(0.95, 1.05)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale)
                    aug = cv2.warpAffine(img, M, (w, h))
                    # 亮度抖动
                    aug = aug.astype(np.float32) * np.random.uniform(0.9, 1.1)
                    aug = np.clip(aug, 0, 255).astype(np.uint8)
                    imgs_aug.append(aug)
                imgs = imgs_aug

            # 应用 transform
            tensor_list = []
            for img in imgs:
                aug = transform(image=img)
                tensor_list.append(aug['image'])
            batch_tensor = torch.stack(tensor_list).to(device)

            logit = model(batch_tensor).squeeze(1)
            prob = torch.sigmoid(logit).cpu().numpy()
            probs_list.append(prob)

        # TTA 平均
        avg_prob = np.mean(probs_list, axis=0)
        batch_probs += avg_prob * weight

    return batch_probs


# ==================== 加载模型（修复 img_size 问题） ====================
def load_model(ckpt_path, model_name, size, device):
    # EfficientNet 系列不能传 img_size，其他模型可以
    create_kwargs = {'pretrained': False, 'num_classes': 1}
    if not any(x in model_name for x in ['efficientnet', 'tf_efficientnet', 'resnet', 'mobilenet']):
        create_kwargs['img_size'] = size

    model = timm.create_model(model_name, **create_kwargs)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    transform = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return model, transform


# ==================== 主程序 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIGC Detection Inference & Evaluation")
    parser.add_argument('--mode', type=str, choices=['eval', 'submission'], default='eval')
    parser.add_argument('--csv', type=str, default=None, help='验证集 csv (image_path,label,...)')
    parser.add_argument('--img_folder', type=str, default=None, help='测试文件夹 (submission 模式)')
    parser.add_argument('--output_csv', type=str, default='submission.csv')

    # 模型配置
    parser.add_argument('--single_ckpt', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--single_size', type=int, default=384)

    parser.add_argument('--ensemble', type=str, nargs='+', default=None,
                        help='格式: ckpt:model_name:size:weight')

    parser.add_argument('--tta', type=int, default=8, help='TTA 次数，1=关闭')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.05, help='分类阈值，设为0自动寻优')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==================== 加载模型 ====================
    models, transforms, weights = [], [], []
    if args.single_ckpt:
        print(f"Loading single model: {args.single_ckpt}")
        m, t = load_model(args.single_ckpt, args.single_model, args.single_size, device)
        models = [m]
        transforms = [t]
        weights = [1.0]
    elif args.ensemble:
        print(f"Loading ensemble ({len(args.ensemble)} models)")
        for item in args.ensemble:
            ckpt, name, sz, w = item.split(':')
            sz, w = int(sz), float(w)
            print(f"  → {os.path.basename(ckpt)} | {name} | {sz}x{sz} | w={w:.2f}")
            m, t = load_model(ckpt, name, sz, device)
            models.append(m)
            transforms.append(t)
            weights.append(w)
        weights = np.array(weights)
        weights = weights / weights.sum()
    else:
        raise ValueError("请指定 --single_ckpt 或 --ensemble")

    # ==================== 数据加载 ====================
    if args.mode == 'eval':
        assert args.csv, "评估模式必须提供 --csv"
        dataset = AIGCDataset(csv_file=args.csv)
    else:
        assert args.img_folder, "submission 模式必须提供 --img_folder"
        dataset = AIGCDataset(img_folder=args.img_folder)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    # ==================== 推理 ====================
    all_probs = []
    all_labels = []
    all_ids = []

    for batch_imgs_np, labels_batch, paths_batch, is_eval_batch in tqdm(dataloader, desc="Inferring"):
        probs = predict_tta_batch(models, transforms, weights, batch_imgs_np, device, tta=args.tta)
        all_probs.extend(probs)

        if args.mode == 'eval':
            all_labels.extend(labels_batch)
        else:
            for p in paths_batch:
                img_id = os.path.splitext(os.path.basename(p))[0]
                all_ids.append(img_id)

    all_probs = np.array(all_probs)

# ==================== 结果输出 ====================
    if args.mode == 'eval':
        labels = np.array(all_labels)
        
        # ---------- 1. 自动寻找最佳阈值 ----------
        if args.threshold <= 0:
            thresholds = np.sort(all_probs)
            best_f1 = 0
            best_th = 0.5
            best_preds = None
            for th in thresholds:
                pred = (all_probs >= th).astype(int)
                f1 = f1_score(labels, pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
                    best_preds = pred.copy()
            args.threshold = best_th
            print(f"\n最佳阈值自动搜索完成: {best_th:.5f} → F1 = {best_f1:.6f}")
        else:
            best_preds = (all_probs >= args.threshold).astype(int)

        # ---------- 2. 计算所有指标 ----------
        preds = best_preds if args.threshold <= 0 else (all_probs >= args.threshold).astype(int)
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, all_probs)
        f1 = f1_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        # ---------- 3. 收集所有样本的路径 ----------
        # 重要：重新收集所有样本的完整路径
        dataset_paths = []
        for idx in range(len(dataset)):
            item = dataset.samples[idx]
            dataset_paths.append(item['path'])
        
        # 确保长度一致
        assert len(dataset_paths) == len(all_probs) == len(labels), f"长度不一致: paths={len(dataset_paths)}, probs={len(all_probs)}, labels={len(labels)}"
        
        # ---------- 4. 保存完整结果（超级有用！） ----------
        result_df = pd.DataFrame({
            'image_path': dataset_paths,
            'prob': all_probs,
            'pred': preds,
            'label': labels
        })
        result_df['error'] = (result_df['pred'] != result_df['label'])
        
        # 按错误率排序，方便查看
        result_df = result_df.sort_values('prob', ascending=False)

        # 保存详细预测结果
        os.makedirs("eval_results", exist_ok=True)
        model_name = args.single_model or f"ensemble_{len(models)}models"
        detail_csv = f"eval_results/detail_{model_name}_{args.tta}tta.csv"
        result_df.to_csv(detail_csv, index=False)

        # 保存摘要报告
        summary = {
            'model': model_name,
            'tta': args.tta,
            'threshold': float(args.threshold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'total_samples': int(len(labels)),
            'best_f1_threshold': float(best_th) if args.threshold <= 0 else None,
            'best_f1': float(best_f1) if args.threshold <= 0 else None,
        }
        
        import json
        with open(f"eval_results/summary_{model_name}_{args.tta}tta.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # ---------- 5. 打印美观结果 ----------
        print("\n" + "="*70)
        print(" " * 20 + "FINAL EVALUATION RESULT")
        print("="*70)
        print(f"Model       : {model_name}")
        print(f"TTA         : {args.tta}x")
        print(f"Threshold   : {args.threshold:.5f} {'(Auto)' if args.threshold <= 0 else ''}")
        print(f"Accuracy    : {acc:.6f}")
        print(f"Precision   : {prec:.6f}")
        print(f"Recall      : {rec:.6f}")
        print(f"F1 Score    : {f1:.6f}  ← ← ← 比赛看这个！")
        print(f"AUC         : {auc:.6f}")
        print(f"Confusion   : TP={tp} | FP={fp} | FN={fn} | TN={tn}")
        print(f"Error Rate  : {(fp + fn)/len(labels)*100:.3f}%  ({fp+fn}/{len(labels)})")
        print("="*70)
        print(f"详细预测已保存 → {detail_csv}")
        print(f"摘要报告已保存 → eval_results/summary_{model_name}_{args.tta}tta.json")
        
        # 显示错误样本统计
        error_df = result_df[result_df['error'] == True]
        print(f"\n错图统计 ({len(error_df)} 张):")
        print(f"  真图被错判为假 (1→0): {len(error_df[(error_df['label']==1) & (error_df['pred']==0)])}")
        print(f"  假图被错判为真 (0→1): {len(error_df[(error_df['label']==0) & (error_df['pred']==1)])}")
        
        print("\n查看错误的命令:")
        print(f"  # 假图错判为真 (0→1):")
        print(f"  awk -F',' '$4==0 && $3==1 {{print $1}}' {detail_csv}")
        print(f"  # 真图错判为假 (1→0):")
        print(f"  awk -F',' '$4==1 && $3==0 {{print $1}}' {detail_csv}")
        print("="*70)