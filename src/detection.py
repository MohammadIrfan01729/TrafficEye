from ultralytics import YOLO
import cv2

helmet_model = YOLO('models/helmet.pt')
vehicle_model = YOLO('models/vehicle.pt')
rider_model = YOLO('models/rider.pt')
plate_model = YOLO('models/plate.pt')


def process_image(img):

    helmet_results = helmet_model(img)[0]
    vehicle_results = vehicle_model(img)[0]
    rider_results = rider_model(img)[0]

    motorcycles, riders, helmets, plates = [], [], [], []

    for box in vehicle_results.boxes:
        if vehicle_model.names[int(box.cls[0])] == "Motorcycle":
            motorcycles.append(box.xyxy[0].cpu().numpy())

    for box in rider_results.boxes:
        riders.append(box.xyxy[0].cpu().numpy())

    for box in helmet_results.boxes:
        helmets.append((box.xyxy[0].cpu().numpy(),
                        helmet_model.names[int(box.cls[0])]))

    for rbox in riders:
        x1,y1,x2,y2 = map(int, rbox)
        crop = img[y1:y2,x1:x2]

        pres = plate_model(crop)[0]

        for pbox in pres.boxes:
            px1,py1,px2,py2 = map(int,pbox.xyxy[0])
            plates.append([px1+x1,py1+y1,px2+x1,py2+y1])

    return motorcycles, riders, helmets, plates