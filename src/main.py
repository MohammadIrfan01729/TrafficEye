import os
import cv2
import pandas as pd
from violation import detect_violations
from reasoning import generate_reasoning

# ---------------- CONFIG ----------------
IMAGE_FOLDER = "data/day_dataset"
OUTPUT_FILE = "output/results.xlsx"

# ---------------------------------------

data = []

# Check folder exists
if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Folder not found: {IMAGE_FOLDER}")
    exit()

images = [f for f in os.listdir(IMAGE_FOLDER) 
          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

print(f"🔍 Found {len(images)} images")

# ---------------- MAIN LOOP ----------------
for idx, img_name in enumerate(images):

    print(f"Processing {idx+1}/{len(images)}: {img_name}")

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"⚠️ Skipping unreadable image: {img_name}")
        continue

    try:
        results = detect_violations(img)
    except Exception as e:
        print(f"❌ Error in detection for {img_name}: {e}")
        continue

    # Skip if no violations
    if not results:
        continue

    for res in results:
        try:
            reasoning_text = generate_reasoning(
                res["violations"],
                res["plate"]
            )
        except Exception as e:
            print(f"⚠️ Reasoning error for {img_name}: {e}")
            reasoning_text = "Reasoning generation failed"

        data.append({
            "image_name": img_name,
            "violation_type": ", ".join(res["violations"]),
            "plate_text": res["plate"],
            "ocr_confidence": res["confidence"],
            "reasoning": reasoning_text
        })

# ---------------- SAVE OUTPUT ----------------
if not data:
    print("⚠️ No violations found in dataset")
else:
    df = pd.DataFrame(data)

    os.makedirs("output", exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    print(f"📊 Total violations detected: {len(data)}")