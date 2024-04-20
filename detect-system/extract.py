import argparse
import time
from pathlib import Path
import os.path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tqdm import tqdm
from bridge_wrapper import YOLOv7_DeepSORT
from detection_helpers import Detector
from collections import defaultdict
import glob  

def load_video_into_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames
    

def extract():
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    print(dir(opt))
    
    detector = Detector()
    detector.load_model(weights=opt.detect_weights, img_size=opt.img_size, trace=not opt.no_trace)
    
    
    # print(dir(opt))
    tracker = YOLOv7_DeepSORT(reID_model_path=opt.reID_weights, detector=detector)
    
    
    
    output_folder = Path(opt.output)    
    os.makedirs(output_folder, exist_ok=True)
    
    
    if os.path.isdir(opt.source):
        videos = glob.glob(f"{opt.source}/*.MP4")
    else:
        videos = [opt.source]
    print("Tracking video")
    for video_name in videos:
        file_name_without_extension = os.path.basename(video_name).split('.')[0]
        output_path = output_folder / file_name_without_extension
        
        os.makedirs(output_path, exist_ok=True)
        
        
        
        
        frame_results_face = tracker.track_video(video_name, output=str(output_path / "labeled.mp4"), show_live = False, skip_frames = 0, count_objects = True, verbose=1)
        video = load_video_into_frames(video_name)
        print("Video has", len(video), "frames")
        
        results = defaultdict(list)
        
        for frame_id, frame_result in enumerate(frame_results_face):
            for type, id, confidence, bbox in frame_result:
                results[f'{type}_{id}'].append([frame_id, bbox])
                
        for key, value in results.items():
            for i in range(len(value) - 1):
                assert value[i][0] <  value[i+1][0], f"Frames are not in order {value[i][0]} and {value[i+1][0]}"
                
        
        frame_results = defaultdict(list)
        
        for key, value in results.items():
            print(f'{key} with {len(value)} frames')
            i = 0
            for frame_id, bbox in value:
                frame = video[frame_id]
                print(frame.shape)
                x1, y1, x2, y2 = bbox
                # to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    print("Empty crop")
                    continue
                # cv2.resize(crop, ())
                os.makedirs(f"{output_path}/{key.replace('person_', '')}", exist_ok=True)
                cv2.imwrite(f"{output_path}/{key.replace('person_', '')}/{i}.jpg", crop)
                
                i = i + 1
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_weights', nargs='+', type=str, default='detect-system/runs/train/exp10/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--reID_weights', type=str, default='detect-system/mars-small128.pb', help='reID model path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', default='out_extract', help='save results to project/name')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        extract()