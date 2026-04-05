import numpy as np
from utils import compute_iou, get_center

def associate(objects, motorcycles):

    mapping = {i: [] for i in range(len(motorcycles))}

    for obj in objects:
        best_match = -1
        best_score = 0

        ox, oy = get_center(obj)

        for i, mbox in enumerate(motorcycles):
            mx, my = get_center(mbox)

            dist = np.sqrt((ox-mx)**2 + (oy-my)**2)
            iou = compute_iou(obj, mbox)

            score = iou + (1/(dist+1e-5))

            if score > best_score:
                best_score = score
                best_match = i

        if best_match != -1:
            mapping[best_match].append(obj)

    return mapping