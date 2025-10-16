import os
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from tqdm import tqdm 


# --- 설정 ---
USE_GPU =  False  #True 
providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if USE_GPU
            else ["CPUExecutionProvider"])
det_size = (640, 640)
det_thresh = 0.5

# --- FaceAnalysis 초기화 ---
app = FaceAnalysis(
    name="buffalo_l",
    root='./insightface',
    allowed_modules=['detection', 'recognition', 'genderage'],
    providers=providers
)
app.prepare(ctx_id=(0 if USE_GPU else -1), det_size=det_size, det_thresh=det_thresh)
swapper = insightface.model_zoo.get_model('./Swap/insightface/models/inswapper_128.onnx', providers=providers)


# --- 폴더 설정 ---
INPUT_DIR = './Dataset/original'
OUTPUT_DIR = './Dataset/swapped_new'


# --- 남자/여자 dummy face 로드 ---
dummy_male_path = './Swap/dummy_faces/dummy_face_male.png'
dummy_female_path = './Swap/dummy_faces/dummy_face_female.png'

img_m = cv2.imread(dummy_male_path)
img_f = cv2.imread(dummy_female_path)
if img_m is None or img_f is None:
    raise ValueError("남자/여자 dummy face 이미지를 불러오지 못했습니다.")

faces_m = app.get(img_m)
faces_f = app.get(img_f)
if not faces_m or faces_m[0].normed_embedding is None:
    raise ValueError("남자 dummy face embedding 추출 실패")
if not faces_f or faces_f[0].normed_embedding is None:
    raise ValueError("여자 dummy face embedding 추출 실패")

face_m = faces_m[0]
face_f = faces_f[0]


# --- input img 로드 & faceswap 수행---
img_files = sorted([f for f in os.listdir(INPUT_DIR)])

for img_file in tqdm(img_files, desc="Processing images"):
    input_img_path = os.path.join(INPUT_DIR, img_file)
    output_img_path = os.path.join(OUTPUT_DIR, img_file)

    img = cv2.imread(input_img_path)
    if img is None:
        print(f"[SKIP] 이미지 로드 실패: {input_img_path}")
        continue

    faces = app.get(img)
    if not faces:
        #df.loc[df['frame'] == img_file, 'faceswap'] = 0
        #shutil.copy(input_img_path, output_img_path)
        continue
    else:
        faceswap_done = 0
        for face in faces:
            if face.normed_embedding is None:
                continue

            # 성별 기반 dummy 선택
            target_face = face_m if face.sex == 'M' else face_f

            try:
                # 얼굴 스왑
                result = swapper.get(img, face, target_face, paste_back=True)
                img = result  # 다음 얼굴도 같은 프레임에서 교체
                faceswap_done = 1
            except Exception as e:
                print(f"[ERROR] 스왑 실패: {input_img_path} → {e}")

        # 최종 이미지 저장 (swap 성공 여부와 관계없이)
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rimg = Image.fromarray(temp)
        rimg.save(output_img_path)
        #print(f"save: {output_img_path} ")

            

