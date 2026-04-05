import numpy as np

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = (x2-x1)*(y2-y1) + (x2g-x1g)*(y2g-y1g) - inter

    return inter/union if union else 0


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, (y1+y2)/2)