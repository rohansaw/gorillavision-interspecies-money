import cv2
import logging
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
import supervision as sv

from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


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
            print("img", img)
            resize_transform = T.Compose([
                T.Resize((960, 1792)),  # Resize to a size that is a multiple of 32
                T.ToTensor()
            ])
            img_tensor = resize_transform(img)

            # Run MegaDetector on the frame
            detection_result = detection_model.single_image_detection(img_tensor)
            detections = detection_result['detections']
            print(detection_result)
            # detection_result['img_id'] = 'test.png'
            # pw_utils.save_detection_images(detection_result, "./demo_output")
            try:
                pw_utils.save_crop_images(detection_result, "crop_output")
            except Exception as e:
                print(e)

            # Extract gorilla bounding boxes from MegaDetector detections
            gorilla_boxes = []
            for idx, class_id in enumerate(detections.class_id):
                if class_id == 0:  # Assuming class_id 0 is for gorillas/animals
                    gorilla_boxes.append(detections.xyxy[idx])

            if len(gorilla_boxes) == 0:
                logging.info("No gorillas detected by MegaDetector in this frame.")
                continue

            # Crop the frame based on the bounding boxes and run YOLOv8 on each crop
            if len(gorilla_boxes) == 0:
                continue
            print(gorilla_boxes)
            for box in gorilla_boxes:
                print(box)
                print(np.array(img.convert("RGB")))
                cropped_img = sv.crop_image(
                    image=np.array(img.convert("RGB")), xyxy=box
                )
                print(cropped_img)
                with sv.ImageSink(target_dir_path="/Users/lukaslaskowski/Documents/HPI/gorillavision/gorillavision-interspecies-money/pipeline/images", overwrite=True) as sink:
                    sink.save_image(
                    image=cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR),
                    image_name="testimage.png"
                    )
                #sv.save_image(image=cropped_img, size=(16, 16))
                sv.plot_image(image=cropped_img, size=(16, 16))

                # cropped_img = only_crop_image(img, *box)
                #x1, y1, x2, y2 = map(int, box)
                # cropped_frame = frame[y1:y2, x1:x2]
                #cropped_tensor = F.crop(img_tensor, top=y1, left=x1, height=y2 - y1, width=x2 - x1)
                
                # image_np = cropped_tensor.permute(1, 2, 0).numpy()
                # plt.imshow(image_np)
                # plt.axis('off')  # Turn off axis for better visualization
                # plt.show()
                break
                if cropped_tensor.size(1) == 0 or cropped_tensor.size(2) == 0:
                    logging.warning(f"Invalid crop for bounding box {box}, skipping.")
                    continue
                cropped_frame = transform(cropped_tensor)
                #cropped_frame = cropped_tensor[0].permute(1, 2, 0).numpy()
                print("writing", cropped_frame)
                cropped_frame.save('./image.png')
                print("written")
                #cv2.imwrite("/Users/lukaslaskowski/Documents/HPI/gorillavision/gorillavision-interspecies-money/image.png", cropped_frame)
                break
                #cv2.imwrite("/Users/lukaslaskowski/Documents/HPI/gorillavision/gorillavision-interspecies-money/image.png", cropped_frame)
                # Run YOLOv8 tracking on the cropped frame
                # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

                results = model.track(cropped_frame, persist=True, conf=float(confidence), vid_stride=int(vid_stride),device="cpu", iou=0.2)
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