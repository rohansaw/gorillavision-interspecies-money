import os
from pathlib import Path
import logging
import cv2
from tqdm import tqdm
from PIL import Image

def calculate_single_bbox(bbox, width, height):
    # try:
    #     logging.info(f"Detected {bbox.cls} with confidence {bbox.conf:.2f}")
    # except:
    #     print("Error")
    # print(bbox)
    box = bbox['xyxy']  # Bounding box coordinates [x1, y1, x2, y2]
    x_center = (box[0] + box[2]) / 2 / width
    y_center = (box[1] + box[3]) / 2 / height
    box_width = (box[2] - box[0]) / width
    box_height = (box[3] - box[1]) / height
    return [int(bbox['cls']), x_center, y_center, box_width, box_height]

def process_image(image_path, model, confidence, vid_stride):
    tracks = {}
    cap = cv2.VideoCapture(str(image_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Processing video {image_path} with {length} frames.")
    i = 0
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
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, conf=float(confidence), vid_stride=int(vid_stride), iou=0.2)
            nothing_in_track = False
            if results[0].boxes.is_track is False:
                nothing_in_track = True
                results = model.predict(frame, conf=float(confidence),vid_stride=int(vid_stride))
                gorilla_detected_in_predict = len(results[0].boxes.cls) > 0
                print("Nothing in track, predicting with predict function", gorilla_detected_in_predict)
            for result in results:
                #print(result.boxes)
                    #tracks['no_track'] = 
                    # predict_results = model.predict(frame, conf=float(confidence))
                    # print("dsl", predict_results[0].boxes)
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
    for key in tracks.keys():
        tracks[key]['bboxes'] = [] #calculate_bbox(tracks[result]['result'])
        for result in tracks[key]['result']:
            tracks[key]['bboxes'].append(calculate_single_bbox(result, Image.fromarray(result['img'], "RGB").size[0], Image.fromarray(result['img'], "RGB").size[1]))
    cap.release()
    return tracks

#def save_results(results, output_path):
    