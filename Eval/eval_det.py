import os, csv
from collections import defaultdict

# ========= 설정 =========
GT_CSV = "./Eval/gt.csv"
ROOT_DIR = "./Eval/det_res"

PRED_FILES = {
    "original": {
        "SCRFD":    os.path.join(ROOT_DIR, "ori_scrdf.csv"),
        "yoloface": os.path.join(ROOT_DIR, "ori_yoloface.csv"),
        "mtcnn":    os.path.join(ROOT_DIR, "ori_mtcnn.csv"),
    },
    "faceswap": {
        "SCRFD":    os.path.join(ROOT_DIR, "swap_scrdf.csv"),
        "yoloface": os.path.join(ROOT_DIR, "swap_yoloface.csv"),
        "mtcnn":    os.path.join(ROOT_DIR, "swap_mtcnn.csv"),
    }
}

MODELS = ["SCRFD", "yoloface", "mtcnn"]
IOU_THRESH = 0.50
MIN_SIDE = 1.0
# =======================

def to_float(s):
    try: return float(str(s).strip())
    except: return None

def norm_box(x1,y1,x2,y2):
    if x1>x2: x1,x2 = x2,x1
    if y1>y2: y1,y2 = y2,y1
    return (x1,y1,x2,y2)

def iou(b1,b2):
    x1,y1,x2,y2 = b1; X1,Y1,X2,Y2 = b2
    ix1,iy1 = max(x1,X1), max(y1,Y1)
    ix2,iy2 = min(x2,X2), min(y2,Y2)
    iw,ih = max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter = iw*ih
    a1 = max(0.0,(x2-x1))*max(0.0,(y2-y1))
    a2 = max(0.0,(X2-X1))*max(0.0,(Y2-Y1))
    denom = a1 + a2 - inter
    return inter/denom if denom>0 else 0.0

def valid_box(b):
    x1,y1,x2,y2 = b
    return (x2-x1)>=MIN_SIDE and (y2-y1)>=MIN_SIDE

def read_gt_csv(path):
    gt = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            img = (row.get("img") or "").strip()
            x1=to_float(row.get("x1")); y1=to_float(row.get("y1"))
            x2=to_float(row.get("x2")); y2=to_float(row.get("y2"))
            if None in (img,x1,y1,x2,y2): continue
            b = norm_box(x1,y1,x2,y2)
            if valid_box(b): gt[img].append(b)
    return gt

def read_pred_csv(path):
    pred = defaultdict(list)
    with open(path, newline='', encoding='utf-8') as f:
        first = f.readline(); f.seek(0)
        has_header = first.lower().startswith("img,")
        if has_header:
            rdr = csv.DictReader(f)
            for row in rdr:
                img = (row.get("img") or "").strip()
                x1=to_float(row.get("x1")); y1=to_float(row.get("y1"))
                x2=to_float(row.get("x2")); y2=to_float(row.get("y2"))
                if None in (img,x1,y1,x2,y2): continue
                if x1==-1 and y1==-1 and x2==-1 and y2==-1: continue
                b = norm_box(x1,y1,x2,y2)
                if valid_box(b): pred[img].append(b)
        else:
            rdr = csv.reader(f)
            for row in rdr:
                if not row or len(row)<5: continue
                img = row[0].strip()
                x1=to_float(row[1]); y1=to_float(row[2])
                x2=to_float(row[3]); y2=to_float(row[4])
                if None in (img,x1,y1,x2,y2): continue
                if x1==-1 and y1==-1 and x2==-1 and y2==-1: continue
                b = norm_box(x1,y1,x2,y2)
                if valid_box(b): pred[img].append(b)
    return pred

def match_greedy(gt_boxes, pred_boxes, thr):
    if not gt_boxes and not pred_boxes: return 0,0,0,[]
    pairs=[]
    for gi,g in enumerate(gt_boxes):
        for pi,p in enumerate(pred_boxes):
            pairs.append((iou(g,p), gi,pi))
    pairs.sort(reverse=True, key=lambda x:x[0])
    used_g, used_p = set(), set()
    matches=[]
    for v,gi,pi in pairs:
        if v<thr: break
        if gi in used_g or pi in used_p: continue
        used_g.add(gi); used_p.add(pi); matches.append((gi,pi,v))
    TP=len(matches)
    FP=max(0,len(pred_boxes)-TP)
    FN=max(0,len(gt_boxes)-TP)
    return TP,FP,FN,matches

def compute_metrics(TP,FP,FN):
    P = TP/(TP+FP) if (TP+FP)>0 else 0.0
    R = TP/(TP+FN) if (TP+FN)>0 else 0.0  # = Face Detection Accuracy(정의)
    F1 = (2*P*R)/(P+R) if (P+R)>0 else 0.0
    ACC = TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0.0
    return P,R,F1,ACC

def evaluate_set(gt_dict, pred_path, thr):
    pred_dict = read_pred_csv(pred_path)
    TP=FP=FN=0
    for img, gt_boxes in gt_dict.items():
        pred_boxes = pred_dict.get(img, [])
        t,f_p,f_n,_ = match_greedy(gt_boxes, pred_boxes, thr)
        TP+=t; FP+=f_p; FN+=f_n
    P,R,F1,ACC = compute_metrics(TP,FP,FN)
    return {"TP":TP,"FP":FP,"FN":FN,"P":P,"R":R,"F1":F1,"ACC":ACC}

# =========================
# 실행: original vs faceswap 한 번에
# =========================
gt = read_gt_csv(GT_CSV)
GT_TOTAL = sum(len(v) for v in gt.values())

results = {ds:{} for ds in ("original","faceswap")}
for ds in ("original","faceswap"):
    for m in MODELS:
        results[ds][m] = evaluate_set(gt, PRED_FILES[ds][m], IOU_THRESH)

#print(f"\n============== Original vs FaceSwap (IoU >= {IOU_THRESH:.2f}) ==============")
#print(f"\n============== Results  ==============")
print(f"GT 총 얼굴 수: {GT_TOTAL:,}, (IoU >= {IOU_THRESH:.2f})\n")

for m in MODELS:
    ro = results["original"][m]; rf = results["faceswap"][m]
    dR   = rf["R"]   - ro["R"]     # Recall(=Face Detection Accuracy) Δ
    dACC = rf["ACC"] - ro["ACC"]   # Accuracy Δ

    print(f"[{m}]")
    print(f"  - original | TP {ro['TP']:5d}  FP {ro['FP']:5d}  FN {ro['FN']:5d}  "
          f"Precision {ro['P']:.3f}  Recall {ro['R']:.3f}  F1 {ro['F1']:.3f}  Accuracy {ro['ACC']:.3f}")
    print(f"  - faceswap | TP {rf['TP']:5d}  FP {rf['FP']:5d}  FN {rf['FN']:5d}  "
          f"Precision {rf['P']:.3f}  Recall {rf['R']:.3f}  F1 {rf['F1']:.3f}  Accuracy {rf['ACC']:.3f}")
    #print(f"  - Δ (faceswap - original):  ΔR={dR:+.3f}  ΔACC={dACC:+.3f}\n")

print("===========================================================================") 

