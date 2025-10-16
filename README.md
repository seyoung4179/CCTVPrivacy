# 🎥 CCTVPrivacy

## 🧠 Overview
CCTVPrivacy aims to enhance privacy protection in CCTV footage through **face swapping**,  
replacing real faces with **synthetic (swapped) faces** while preserving overall recognition performance and visual naturalness. 

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



It aims to analyze how face anonymization techniques affect recognition and detection accuracy.



## Contact
This page and files are still updating.
Seyoung Jin (22sysy@g.skku.edu), Sungkyunkwan University 

