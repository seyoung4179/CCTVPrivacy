import os, cv2, numpy as np
import pandas as pd
import insightface

# =========================
# 설정
# =========================
ORI_ROOT   = "./Dataset/original"
SWAP_ROOT  = "./Dataset/swapped"
GT_CSV     = "./Eval/gt.csv"     # 컬럼: img,x1,y1,x2,y2
OUT_CSV    = "./Eval/similar_res/similarities_res.csv"  # 유사도/매칭 로그 저장 파일
THRESHOLDS = [0.5, 0.7, 0.9]

CTX_ID = -1  # GPU=0, CPU=-1

# =========================
# 모델 로드: buffalo_l (탐지+정렬+임베딩)
# =========================
app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=CTX_ID, det_size=(640, 640))
print("[OK] buffalo_l 로드 완료")

# =========================
# 유틸
# =========================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    ba = (bx2 - bx1) * (by2 - by1)
    return inter / float(aa + ba - inter + 1e-12)

def bbox_from_face(face):
    # buffalo_l face.bbox: (x1,y1,x2,y2) float
    x1, y1, x2, y2 = map(float, face.bbox)
    return [x1, y1, x2, y2]

def pick_face_by_iou(faces, gt_box, iou_thr=0.0):
    """전체 이미지에서 탐지된 faces 중 GT와 IoU가 가장 큰 얼굴 선택.
       iou_thr는 최소 허용값(0.0이면 어떤 얼굴이든 베스트 1개 선택)"""
    if not faces:
        return None, 0.0
    best, best_iou = None, 0.0
    for f in faces:
        fb = bbox_from_face(f)
        v = iou_xyxy(gt_box, fb)
        if v > best_iou:
            best, best_iou = f, v
    if best is None:
        return None, 0.0
    if best_iou < iou_thr:
        return None, best_iou
    return best, best_iou

def l2n(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def cosine(u, v):
    return float(np.dot(u, v))

# =========================
# GT 로드 & 전처리
# =========================
gt = pd.read_csv(GT_CSV)
for c in ["x1","y1","x2","y2"]:
    gt[c] = pd.to_numeric(gt[c], errors="coerce")
gt = gt.dropna(subset=["img","x1","y1","x2","y2"])
gt = gt[~((gt["x1"]==-1)&(gt["y1"]==-1)&(gt["x2"]==-1)&(gt["y2"]==-1))].copy()

# =========================
# 메인 루프: 전체 이미지 -> 탐지 -> GT와 IoU 최댓값 얼굴 선택 -> 임베딩 -> 유사도
# =========================
rows = []   # 저장: 한 행 = (img, gt_x1..gt_y2, iou_ori, iou_swap, cos_sim)
skipped = 0

for _, r in gt.iterrows():
    img_name = str(r["img"]).strip()
    gx1, gy1, gx2, gy2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
    gt_box = [gx1, gy1, gx2, gy2]

    # 원본/스왑 이미지 로드
    ori_path  = os.path.join(ORI_ROOT,  img_name)
    swap_path = os.path.join(SWAP_ROOT, img_name)
    ori_img   = cv2.imread(ori_path)
    swap_img  = cv2.imread(swap_path)
    if ori_img is None or swap_img is None:
        skipped += 1
        continue

    # 전체 이미지에서 얼굴 탐지 + 정렬 + 임베딩
    faces_o = app.get(ori_img)
    faces_s = app.get(swap_img)

    # GT와 IoU가 가장 큰 얼굴 선택(최소 IoU 임계값은 0.0; 필요시 0.3~0.5로 올려도 됨)
    f_o, iou_o = pick_face_by_iou(faces_o, gt_box, iou_thr=0.0)
    f_s, iou_s = pick_face_by_iou(faces_s, gt_box, iou_thr=0.0)
    if (f_o is None) or (f_s is None):
        skipped += 1
        continue

    # buffalo_l는 이미 정렬 임베딩 제공: face.embedding (512)
    emb_o = l2n(f_o.embedding)
    emb_s = l2n(f_s.embedding)
    sim   = cosine(emb_o, emb_s)

    rows.append([img_name, gx1, gy1, gx2, gy2, iou_o, iou_s, sim])

# =========================
# 결과 집계/저장
# =========================
df = pd.DataFrame(rows, columns=["img","gt_x1","gt_y1","gt_x2","gt_y2","iou_ori","iou_swap","cosine_sim"])
# CSV 저장 (추후 재분석용)
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"[SAVE] 유사도 로그 저장: {OUT_CSV}")


print("===========================================================================") 
print(f"[INFO] 쌍 총계: {len(df)} (스킵 {skipped})")
# TAR 계산 (cosine >= θ 비율)
for th in THRESHOLDS:
    tar = (df["cosine_sim"] >= th).mean() if len(df) > 0 else 0.0
    ok  = int((df["cosine_sim"] >= th).sum())
    print(f"@ cos≥{th:.1f}: {tar:.3f}  ({ok}/{len(df)})")


