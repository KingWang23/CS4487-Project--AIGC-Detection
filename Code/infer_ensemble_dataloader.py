# 支持单模型 / 多模型集成 + TTA

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
from typing import List, Tuple


# ==================== Dataset ====================
class InferenceDataset(Dataset):
    def __init__(self, img_paths: List[str], transforms_list: List[A.Compose]):
        """
        transforms_list: 每个模型对应的 transform（长度 == 模型数量）
        """
        self.img_paths = img_paths
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        filename = os.path.basename(img_path)
        img_id = os.path.splitext(filename)[0]

        # 读取原始图像（RGB）
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 为每个模型准备一份已经 transform 好的 tensor（不做 TTA）
        # 这里只做 Resize + Normalize，TTA 在外面单独处理
        tensors = []
        for transform in self.transforms_list:
            aug = transform(image=img)
            tensor = aug["image"]               # (C,H,W)
            tensors.append(tensor)

        return {
            "img_path": img_path,
            "img_id": img_id,
            "raw_img": img,                     # 用于后续 TTA
            "tensors": tensors                  # 每个模型的基础 tensor（不翻转）
        }


# ==================== TTA 函数 ====================
@torch.no_grad()
def apply_tta(model: nn.Module,
              raw_img: np.ndarray,
              base_transform: A.Compose,
              device: torch.device,
              tta: int = 8) -> float:
    """
    对单张原始图像做 TTA，返回平均概率
    """
    model.eval()
    probs = []

    for i in range(tta):
        if i == 0:
            aug_img = raw_img.copy()
        elif i == 1:
            aug_img = cv2.flip(raw_img, 1)                                   # 水平翻转
        elif i == 2:
            aug_img = cv2.flip(raw_img, 0)                                   # 垂直翻转
        elif i == 3:
            aug_img = cv2.flip(raw_img, -1)                                  # 水平+垂直
        elif i == 4:
            aug_img = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
        elif i == 5:
            aug_img = cv2.rotate(raw_img, cv2.ROTATE_180)
        elif i == 6:
            aug_img = cv2.rotate(raw_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 更多几何 TTA 可自行补充...

        # 随机小扰动（强烈推荐）
        elif 10 <= i < 18:
            angle = np.random.uniform(-12, 12)
            scale = np.random.uniform(0.92, 1.08)
            h, w = raw_img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
            aug_img = cv2.warpAffine(raw_img, M, (w, h))

            # 轻微平移 + 亮度抖动
            tx = np.random.uniform(-0.08, 0.08) * w
            ty = np.random.uniform(-0.08, 0.08) * h
            M_t = np.float32([[1, 0, tx], [0, 1, ty]])
            aug_img = cv2.warpAffine(aug_img, M_t, (w, h))
            aug_img = aug_img.astype(np.float32) * np.random.uniform(0.85, 1.15)
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

        # 多尺度（强烈推荐）
        elif i == 20:
            h, w = raw_img.shape[:2]
            aug_img = cv2.resize(raw_img, (int(w*0.8), int(h*0.8)))
            aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        elif i == 21:
            h, w = raw_img.shape[:2]
            aug_img = cv2.resize(raw_img, (int(w*1.15), int(h*1.15)))
            aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            aug_img = raw_img.copy()

        # 应用基础 transform（Resize + Normalize）
        transformed = base_transform(image=aug_img)
        tensor = transformed["image"].unsqueeze(0).to(device)   # (1,C,H,W)

        logit = model(tensor).squeeze(1)
        prob = torch.sigmoid(logit).cpu().item()
        probs.append(prob)

    return float(np.mean(probs))


# ==================== 加载模型 ====================
def create_model_and_transform(ckpt_path: str, model_name: str, size: int, device) -> Tuple[nn.Module, A.Compose]:
    # 统一模型名称
    if 'efficientnet_b5' in model_name:
        model_name = 'tf_efficientnet_b5'
    elif 'efficientnet_b7' in model_name:
        model_name = 'tf_efficientnet_b7'

    model = timm.create_model(model_name, pretrained=False, num_classes=1, drop_path_rate=0.0)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # 去掉 DataParallel 前缀
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


# ==================== 主程序 ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIGC Detection Inference (DataLoader + Ensemble + TTA)")
    parser.add_argument('--img_folder', type=str, required=True, help='测试图片文件夹')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='输出文件')

    # 单模型模式
    parser.add_argument('--single_ckpt', type=str, default=None)
    parser.add_argument('--single_model', type=str, default=None)
    parser.add_argument('--single_size', type=int, default=384)

    # 集成模式（推荐）
    parser.add_argument('--ensemble', type=str, nargs='+',
                        help='格式: ckpt_path:model_name:size:weight   多组用空格分隔')

    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--tta', type=int, default=1, help='TTA 次数，推荐 12~24')
    parser.add_argument('--batch_size', type=int, default=1, help='DataLoader batch_size，TTA 模式下建议 1')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------- 加载所有模型 -------------------
    models = []
    transforms = []
    weights = []

    if args.single_ckpt:
        print(f"Loading single model: {args.single_ckpt}")
        model, transform = create_model_and_transform(args.single_ckpt, args.single_model,
                                                      args.single_size, device)
        models.append(model)
        transforms.append(transform)
        weights.append(1.0)
    elif args.ensemble:
        print(f"Loading {len(args.ensemble)} models for ensemble...")
        for item in args.ensemble:
            ckpt, name, sz, w = item.split(':')
            sz = int(sz)
            w = float(w)
            print(f"  → {os.path.basename(ckpt)} | {name} | {sz}x{sz} | weight={w:.3f}")
            model, transform = create_model_and_transform(ckpt, name, sz, device)
            models.append(model)
            transforms.append(transform)
            weights.append(w)
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()
    else:
        raise ValueError("请指定 --single_ckpt 或 --ensemble")

    # ------------------- 图片路径（严格字母序） -------------------
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG')
    img_paths = [os.path.join(args.img_folder, f) for f in os.listdir(args.img_folder)
                 if f.lower().endswith(exts)]
    img_paths.sort()
    print(f"Found {len(img_paths)} images")

    # ------------------- Dataset & DataLoader -------------------
    dataset = InferenceDataset(img_paths, transforms)   # transforms 列表长度 = 模型数量
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=False)

    # ------------------- 推理 -------------------
    results = []
    pbar = tqdm(dataloader, desc="Inferring", total=len(dataloader))

    for batch in pbar:
        raw_imgs = batch["raw_img"]          # List[np.ndarray]
        img_ids = batch["img_id"]

        # 每张图分别做 TTA（batch, TTA 次数相同）
        for idx in range(len(img_ids)):
            cur_probs = []
            for model_idx, (model, transform) in enumerate(zip(models, transforms)):
                prob = apply_tta(model,
                                 raw_imgs[idx].numpy() if torch.is_tensor(raw_imgs[idx]) else raw_imgs[idx],
                                 transform,
                                 device,
                                 tta=args.tta)
                cur_probs.append(prob * weights[model_idx])

            final_prob = sum(cur_probs)
            label = 1 if final_prob > args.threshold else 0
            results.append({"ID": img_ids[idx], "label": int(label)})

    # ------------------- 保存 -------------------
    df = pd.DataFrame(results)
    df = df.sort_values("ID").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nInference finished! Submission saved to: {args.output_csv}")
    print(f"  AI-generated (1): {df['label'].sum()}   |   Real (0): {len(df) - df['label'].sum()}")