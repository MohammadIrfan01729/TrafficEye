from detection import process_image
from association import associate
from utils import compute_iou
from ocr import extract_plate_text

def detect_violations(img):

    motorcycles, riders, helmets, plates = process_image(img)

    rider_map = associate(riders, motorcycles)
    plate_map = associate(plates, motorcycles)

    results = []

    for i, mbox in enumerate(motorcycles):

        violations = []
        associated_riders = rider_map[i]
        associated_plates = plate_map[i]

        for rbox in associated_riders:
            has_helmet = False

            for hbox, hcls in helmets:
                if compute_iou(rbox, hbox) > 0.2:
                    if hcls == "Helmet":
                        has_helmet = True
                    else:
                        violations.append("No Helmet")

            if not has_helmet:
                violations.append("No Helmet")

        if len(associated_riders) >= 3:
            violations.append("Triple Riding")

        if len(associated_plates) == 0:
            violations.append("No Plate")
            plate_text, conf = "No Plate", 0.0
        else:
            px1,py1,px2,py2 = map(int, associated_plates[0])
            crop = img[py1:py2, px1:px2]

            plate_text, conf = extract_plate_text(crop)

        if violations:
            results.append({
                "violations": list(set(violations)),
                "plate": plate_text,
                "confidence": conf
            })

    return results