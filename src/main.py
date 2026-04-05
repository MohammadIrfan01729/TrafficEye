import os
import cv2
import pandas as pd
from violation import detect_violations
from reasoning import generate_reasoning

image_folder = "data/day_dataset"

data = []

for img_name in os.listdir(image_folder):

    if not img_name.endswith(('.jpg','.png','.jpeg')):
        continue

    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)

    results = detect_violations(img)

    for res in results:
        data.append({
            "image_name": img_name,
            "violation_type": ", ".join(res["violations"]),
            "plate_text": res["plate"],
            "ocr_confidence": res["confidence"],
            "reasoning": reasoning
        })

df = pd.DataFrame(data)
df.to_excel("output/results.xlsx", index=False)

print("✅ Done")