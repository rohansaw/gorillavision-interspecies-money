import os
from pathlib import Path
import logging
import cv2
from tqdm import tqdm
from PIL import Image

def calculate_bbox(results, widths, heights):
    images = []
    for idx, result in enumerate(results):
        data = []
        for bbox in result.boxes:
            try:
                logging.info(f"Detected {bbox.cls} with confidence {bbox.conf:.2f}")
            except:
                print("Error")
            box = bbox.xyxy.tolist()[0]  # Bounding box coordinates [x1, y1, x2, y2]
            x_center = (box[0] + box[2]) / 2 / widths[idx]
            y_center = (box[1] + box[3]) / 2 / heights[idx]
            box_width = (box[2] - box[0]) / widths[idx]
            box_height = (box[3] - box[1]) / heights[idx]
            data.append([int(bbox.cls.item()), x_center, y_center, box_width, box_height])
        images.append({'bboxes': data, 'img': result.orig_img})
    return images

def process_image(image_path, model, confidence, vid_stride):
    results = model(image_path, conf=float(confidence), vid_stride=int(vid_stride))
    widths = []
    heights = []
    for result in results:
        widths.append(Image.fromarray(result.orig_img, "RGB").size[0])
        heights.append(Image.fromarray(result.orig_img, "RGB").size[1])
    #imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image = cv2.imread(str(image_path))
    #height, width = image.shape[:2]
    return calculate_bbox(results, widths, heights)
    #return data

#def save_results(results, output_path):
    