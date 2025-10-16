# 🎥 CCTVPrivacy
CCTVPrivacy protects personal identity in CCTV footage through **face swapping**,  
replacing original faces with **synthetic ones** while keeping scenes visually realistic.

## Key Features
- Detects faces in input CCTV images.  
- Replaces detected faces with synthetic ones using deep learning-based face swapping.  
- Evaluates the impact of face swapping on face detection performance and privacy preservation.

## 🧩 Model Downloads
### 🔵 InsightFace (Face Recognition / Face Swap)
**InsightFace** is used for both **face recognition** and **face swapping**.  
Download the official pre-trained models from:  
🔗 [https://github.com/deepinsight/insightface/releases](https://github.com/deepinsight/insightface/releases)

> Recommended model: **buffalo_l**  
> Files include: `w600k_r50.onnx`, `genderage.onnx`, `det_10g.onnx`

Place the downloaded files in: /CCTVPrivacy/Swap/insightface/models/buffalo_l/

### 🧩 Folder Descriptions

| Folder | Description |
|--------|--------------|
| **Dataset/** | Contains input CCTV images and their corresponding face-swapped results. |
| **Swap/** | Implements the **Face Swap** process using InsightFace or similar models — detecting, aligning, and replacing faces with synthetic ones. |
| **Face_recognition/** | Provides functionality for face feature extraction, embedding comparison, and recognition using pretrained models. |
| **Eval/** | Includes scripts to analyze how Face Swap techniques influence face recognition performance and privacy preservation. |


## Contact
This page and files are still updating.
Seyoung Jin (22sysy@g.skku.edu), Sungkyunkwan University 

