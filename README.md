# CCTVPrivacy
# 🎥 CCTVPrivacy — Face Swapping & Face Detection Evaluation

## 🧠 Overview
**CCTVPrivacy** is a project designed to evaluate the performance of **Face Detection** and **Face Swapping** methods  
for **privacy protection in CCTV footage**.  
It aims to analyze how face anonymization techniques affect recognition and detection accuracy.

## 🧩 Project Structure

CCTVPrivacy/
├── Dataset/ # Original and swapped face data
├── Eval/ # Detection evaluation (mAP, Precision, Recall, etc.)
├── Face_recognition/ # Face embedding and recognition modules
├── Swap/ # Face Swapping modules

## 🧩 Model Downloads

### 🟢 YOLOFace (Face Detection)
This project uses **YOLOv8-based Face Detection** models.  
Download the model from the following repository:  
🔗 [https://github.com/lindevs/yolov8-face](https://github.com/lindevs/yolov8-face)

Place the downloaded model in the following folder: /Face_recognition/



### 🔵 InsightFace (Face Recognition / Face Swap)
**InsightFace** is used for both **face recognition** and **face swapping**.  
Download the official pre-trained models from:  
🔗 [https://github.com/deepinsight/insightface/releases](https://github.com/deepinsight/insightface/releases)

> Recommended model: **buffalo_l**  
> Files include: `w600k_r50.onnx`, `genderage.onnx`, `det_10g.onnx`

Place the downloaded files in: /CCTVPrivacy/Swap/insightface/models/buffalo_l/
