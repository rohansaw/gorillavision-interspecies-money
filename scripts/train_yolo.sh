WANDB_MODE=offline python3 train_detection.py \
    --weights /workspace/detect-system/yolov7x.pt \
    --data /workspace/detect-system/cfg/yolo_combined.yaml \
    --hyp /workspace/detect-system/cfg/yolo_hyp.yaml \
    --workers 14 \
    --device 0 \
    --batch-size 32 \
    --epochs 300