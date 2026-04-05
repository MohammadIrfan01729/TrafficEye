from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(use_angle_cls=False, lang='en')

def extract_plate_text(crop):

    crop = cv2.resize(crop, None, fx=2, fy=2)

    result = ocr.ocr(crop)

    if result is None or len(result[0]) == 0:
        return "Unreadable", 0.0

    texts = []
    scores = []

    for line in result[0]:
        text = line[1][0]
        score = line[1][1]

        texts.append(text)
        scores.append(score)

    final_text = " ".join(texts)
    avg_score = sum(scores)/len(scores)

    return final_text, avg_score