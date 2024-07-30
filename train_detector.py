from ultralytics import YOLOv10

PATH_TO_WEIGHTS = ""
PATH_TO_TRAIN_DATA_CONFIG = ""
epochs = 500
batch = 256
imgsz = 640

model = YOLOv10.from_pretrained('jameslahm/yolov10n') if PATH_TO_WEIGHTS is None else YOLOv10(PATH_TO_WEIGHTS)
model.train(data=PATH_TO_TRAIN_DATA_CONFIG, epochs=epochs, batch=batch, imgsz=imgsz)
