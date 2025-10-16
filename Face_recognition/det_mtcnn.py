import os, csv
from glob import glob
from facenet_pytorch import MTCNN
from PIL import Image
import torch, cv2
from tqdm import tqdm 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(keep_all=True, device=device)


INPUT_DIR = "./Dataset/swapped"
Results_FILE = "./Eval/det_res/swap_mtcnn.csv"

img_list = sorted([f for f in os.listdir(INPUT_DIR)])

results = []

for img_name in tqdm(img_list, desc="Processing images"):
    # if img_name == "frame_00386.jpg":
    #    break

    in_path = os.path.join(INPUT_DIR, img_name)
    bgr = cv2.imread(in_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    boxes, probs = detector.detect(pil)

    if boxes is None:
        results.append([img_name, -1, -1, -1, -1, 0.0])
    else:
        for box, p in zip(boxes, probs):
            #if p > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            results.append([img_name, x1, y1, x2, y2, p])
    #break
    

with open(Results_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["img", "x1", "y1", "x2", "y2", "p"])
    writer.writerows(results)


