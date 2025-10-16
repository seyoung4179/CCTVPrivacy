import torch
import cv2
import os
import csv
from tqdm import tqdm
from ultralytics import YOLO

# 1) 디바이스 설정
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('using device =>', device)

# 2) YOLOv8 얼굴 전용 모델 로드 (예: yolov8n-face.pt)
model = YOLO("./Face_recognition/yolov8n-face-lindevs.pt")

img_dir = "./Dataset/swapped"
results_file = "./Eval/det_res/swap_yoloface.csv"

img_list = sorted(os.listdir(img_dir))

# CSV 헤더: 이미지명, 얼굴 인덱스, 좌표, conf
header = ["img", "x1", "y1", "x2", "y2", "conf"]
rows = []

for img in tqdm(img_list, desc="Processing images"):
    img_path = os.path.join(img_dir, img)
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        rows.append([img, None, None, None, None, None])
        continue

    # 추론
    res = model.predict(
        source=img_bgr,
        conf=0.5,
        iou=0.45,
        device=device,
        verbose=False
    )
    r = res[0]

    # 얼굴 없으면 빈행 남기기(선택)
    if r.boxes is None or len(r.boxes) == 0:
        rows.append([img, -1, -1, -1, -1, 0.0])
    else:
        for face_idx, box in enumerate(r.boxes, start=1):  # start=0으로 바꾸면 0부터 시작
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            rows.append([img, x1, y1, x2, y2, conf])

# CSV 저장
with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("결과 CSV 저장:", results_file)

