WANDB_MODE=offline python3 train_detection.py \
    --weights /data/models/yolov7_face_model.pt \
    --data /workspace/reid-system/cfg/yolo_data.yaml \
    --hyp /workspace/reid-system/cfg/yolo_hyp.yaml \
    --workers 8 \
    --device 0