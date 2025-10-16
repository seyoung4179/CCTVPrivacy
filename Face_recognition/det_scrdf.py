import os
import csv
from tqdm import tqdm
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import json

# ===== 설정 =====
INPUT_DIR = "./Dataset/swapped"
Results_FILE = "./Eval/det_res/swap_scrdf.csv"
USE_GPU = True                  # GPU 사용 시 True (onnxruntime-gpu 설치 필요)
MODEL_PACK = "buffalo_l"         

DET_SIZE = (640, 640) 
DET_THRESH = 0.5 

def draw_boxes_and_save(img_bgr, dets, out_path):
    canvas = img_bgr.copy()
    for idx, face in enumerate(dets):
        x1, y1, x2, y2 = face.bbox.astype(int)
        # 경계 보정 & 유효성 체크
        h, w = canvas.shape[:2]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)  # ← canvas에 그림
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main():
    # InsightFace 초기화
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU
                 else ["CPUExecutionProvider"])
    app = FaceAnalysis(name=MODEL_PACK, providers=providers)
    app.prepare(ctx_id=(0 if USE_GPU else -1), det_size=DET_SIZE)

    img_list = sorted([f for f in os.listdir(INPUT_DIR)])
    results = []


    for img_name in tqdm(img_list, desc="Processing images"):
        # if img_name == "frame_00386.jpg":
        #     break
        in_path = os.path.join(INPUT_DIR, img_name)

        img_bgr = cv2.imread(in_path) 
        dets = app.get(img_bgr)

        if len(dets) == 0: 
            results.append([img_name, -1, -1, -1, -1, -1, 0.0])
        else:
            for idx, d in enumerate(dets):
                x1, y1, x2, y2 = map(int, d.bbox)
                score = float(d.det_score)
                results.append([img_name, idx, x1, y1, x2, y2, score])
            #draw_boxes_and_save(img_bgr, dets, out_path)
            
    with open(Results_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img", "id", "x1", "y1", "x2", "y2", "score"])
        writer.writerows(results)


if __name__ == "__main__":
    main()

