from ultralytics import YOLO

model = YOLO("yolov8x.yaml")

results = model.train(data='tt100k.yaml', epochs=490, batch=36, device=[0, 1, 2, 3])
