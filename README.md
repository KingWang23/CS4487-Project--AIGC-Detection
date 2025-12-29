# CS4487 Project -- ELDA: Ensemble Learning with Domain-Specific Data Augmentation for AIGC Image Detection
This repo contains a reference implementation of the work "ELDA: Ensemble Learning with Domain-Specific Data Augmentation for AIGC Image Detection, developed as a project for CS4487 Machine Learning course (Semester A, 2025-2026) at the City University of Hong Kong. It achieved a **perfect score (full marks)** and **ranked 1st** in the class on the reserved test set.

# Abstract 
The rapid advancement of AI-Generated Content (AIGC) technologies has made synthetic images nearly indistinguishable from real photographs, posing significant risks of misinformation and deepfakes. Although numerous state-of-the-art detectors (e.g., CO-SPY, ForgeLens, TruFor, LEGION) achieve impressive results on public benchmarks, they consistently underperform on the private dataset of the **CS4487 AIGC Detection Project**, often falling below even simple Resnet baselines.

In this work, we propose **ELDA** (Ensemble Learning with Domain-specific Data Augmentation), a practical yet extremely effective framework. By integrating four powerful pretrained backbones and introducing aggressive domain-specific augmentations—including DCT-based high-frequency boosting and realistic camera artifact simulation—ELDA not only achieves **99.96% accuracy** on the hidden Kaggle test set, but also attains a perfect **overall score of 1.0** on the reserved test set, securing the **1st rank** in the class.

# Train:
Main Train: train_aigc_detector_without EMA.py 
```shell
1. ConvNeXtV2-Large-384
python train_aigc_detector.py --train_csv dataset_csv/train.csv --val_csv dataset_csv/val.csv --model convnextv2_large.fcmae_ft_in22k_in1k_384 --batch_size 40

2. EfficientNet-B7-384
python train_aigc_detector.py --train_csv dataset_csv/train.csv --val_csv dataset_csv/val.csv --model tf_efficientnet_b7.ns_jft_in1k --batch_size 48

3. EfficientNet-B5-512
python train_aigc_detector.py --train_csv dataset_csv/train.csv --val_csv dataset_csv/val.csv --model tf_efficientnet_b5.ns_jft_in1k  --batch_size 24

4. DeiT3-Large-384
python train_aigc_detector.py --train_csv dataset_csv/train.csv --val_csv dataset_csv/val.csv --model deit3_large_patch16_384.fb_in22k_ft_in1k --batch_size 32
```
# Test
infer_ensemble can be used for both integrated and individual model testing, allowing you to test images within a folder.
```shell
python infer_ensemble.py \
    --img_folder /home/stf/CourseWork/CS4487/Project/AIGC-Detection-Test-1 \
    --single_ckpt /home/stf/CourseWork/CS4487/Project/GPT_1123/four_method/checkpoints/tf_efficientnet_b5_best.pth\
    --single_model tf_efficientnet_b5.ns_jft_in1k \
    --single_size 512 \
    --output_csv tf_efficientnet_b5.csv

python infer_ensemble.py \
    --img_folder test_images/ \
    --ensemble \
        checkpoints/convnextv2_384_best.pth:convnextv2_large.fcmae_ft_in22k_in1k_384:384:1.2 \
        checkpoints/effnetb7_384_best.pth:tf_efficientnet_b7.ns_jft_in1k:384:1.0 \
        checkpoints/effnetb5_512_best.pth:tf_efficientnet_b5.ns_jft_in1k:512:0.9 \
        checkpoints/deit3_384_best.pth:deit3_large_patch16_384.fb_in22k_ft_in1k:384:0.8 \
    --threshold 0.5 \
    --output_csv submission_ensemble.csv
```
infer_new  can generate several images with the highest uncertainty.

infer_with uncentainty can perform val.
```shell
python infer_csv.py \
    --csv dataset_csv/val.csv \
    --single_ckpt checkpoints/tf_efficientnet_b5_best_1123.pth \
    --single_model tf_efficientnet_b5.ns_jft_in1k \
    --single_size 384 \
    --tta 8
```
