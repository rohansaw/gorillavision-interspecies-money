import os
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 1422907118549761
import logging
import cv2

def read_label(labels, id="0"):
    for label in labels:
        if label.split(" ")[0] == id:
            corners = line.split()[1:]
            return [float(val) for val in corners]
    return None

def yolobbox2bbox(x, y, w, h, img_w, img_h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h

def crop_image(img, labels):
    logging.info(f"Reading label {labels}")
    logging.warning(f"Currently always selecting the first gorilla on the image")
    print(labels)
    label = labels[1:]
    print(img)
    imageRGB = cv2.cvtColor(img['img'], cv2.COLOR_BGR2RGB)
    img = Image.fromarray(imageRGB, 'RGB')
    bbox = yolobbox2bbox(*label, img.width, img.height)
    print(bbox)
    cropped_image = img.crop(tuple(bbox))
    return cropped_image

def only_crop_image(image, x, y, w, h):
    # imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = Image.fromarray(imageRGB, 'RGB')
    bbox = yolobbox2bbox(x,y,w,h, image.width, image.height)
    return image.crop(tuple(bbox))