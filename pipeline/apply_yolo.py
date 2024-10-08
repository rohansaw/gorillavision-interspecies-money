import cv2
import logging
from PytorchWildlife.models import detection as pw_detection
from PIL import Image
import torchvision.transforms as T
import numpy as np
from cropper import only_crop_image

def process_image(image_path, model, confidence, vid_stride):
    tracks = {}
    cap = cv2.VideoCapture(str(image_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video {image_path} with {length} frames.")
    i = 0

    # Initialize MegaDetector model
    detection_model = pw_detection.MegaDetectorV5() # Model weights are automatically downloaded.

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if i % int(vid_stride) != 0:
            i += 1
            continue
        i += 1
        logging.info(f"Processing frame {i}")

        if success:
            # Convert frame to PIL Image for MegaDetector
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            resize_transform = T.Compose([
                T.Resize((960, 1792)),  # Resize to a size that is a multiple of 32
                T.ToTensor()
            ])
            img_tensor = resize_transform(img)

            # Run MegaDetector on the frame
            detection_result = detection_model.single_image_detection(img_tensor)
            detections = detection_result['detections']

            # Extract gorilla bounding boxes from MegaDetector detections
            gorilla_boxes = []
            for idx, class_id in enumerate(detections.class_id):
                if class_id == 0:  # Assuming class_id 0 is for gorillas/animals
                    gorilla_boxes.append(detections.xyxy[idx])

            if len(gorilla_boxes) == 0:
                logging.info("No gorillas detected by MegaDetector in this frame.")
                continue

            # Crop the frame based on the bounding boxes and run YOLOv8 on each crop
            for box in gorilla_boxes:
                # cropped_img = only_crop_image(img, *box)
                x1, y1, x2, y2 = map(int, box)
                cropped_frame = frame[y1:y2, x1:x2]
                cv2.imwrite("./image.png", cropped_frame)

                # Run YOLOv8 tracking on the cropped frame
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                results = model.track(cropped_frame, persist=True, conf=float(confidence), vid_stride=int(vid_stride),device=1, iou=0.2)
                nothing_in_track = False
                if results[0].boxes.is_track is False:
                    nothing_in_track = True
                    results = model.predict(cropped_frame, conf=float(confidence), vid_stride=int(vid_stride))
                    gorilla_detected_in_predict = len(results[0].boxes.cls) > 0
                    print("Nothing in track, predicting with predict function", gorilla_detected_in_predict)

                for result in results:
                    if result.boxes.id is not None:
                        for idx, id in enumerate(result.boxes.id):
                            id = "no_track" if nothing_in_track and gorilla_detected_in_predict else str(int(id.item()))
                            print("JJJJ", id == "no_track")
                            if id not in tracks:
                                tracks[id] = {}
                                tracks[id]['result'] = []
                            tracks[id]['result'].append({'xyxy': result.boxes.xyxy[idx].tolist(), 'img': result.orig_img, 'cls': result.boxes.cls[idx].item()})
        else:
            break