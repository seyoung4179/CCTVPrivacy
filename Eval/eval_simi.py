import pandas as pd
import csv

PRED_FILES    = "./Eval/similar_res/similarities_res.csv"  # 유사도/매칭 로그 저장 파일
THRESHOLDS = [0.5, 0.7, 0.9]


df = pd.read_csv(PRED_FILES)

print("===========================================================================") 
print(f"[INFO] 쌍 총계: {len(df)}")
# TAR 계산 (cosine >= θ 비율)
for th in THRESHOLDS:
    tar = (df["cosine_sim"] >= th).mean() if len(df) > 0 else 0.0
    ok  = int((df["cosine_sim"] >= th).sum())
    print(f"@ cos≥{th:.1f}: {tar:.3f}  ({ok}/{len(df)})")


