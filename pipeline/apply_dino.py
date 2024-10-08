import os
import json

from transformers import Dinov2Config, Dinov2ForImageClassification
import safetensors.torch

import cv2
import torch
from torchvision.transforms import v2

# from detection.detection_types import ReIdResult

def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=v2.InterpolationMode.BICUBIC,
    crop_size: int = 224,
) -> v2.Compose:
    transforms_list = [
        v2.Resize(resize_size, interpolation=interpolation),
        v2.CenterCrop(crop_size),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return v2.Compose(transforms_list)

class Dino:
    def __init__(self, model_path, device):
        config_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.safetensors")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = Dinov2Config(**config_dict)

        self.model = Dinov2ForImageClassification(config)
        self.model.eval()
        state_dict = safetensors.torch.load_file(weights_path)
        self.model.eval()
        self.model.load_state_dict(state_dict, strict=False)
        self.id2labels = config.id2label

    def predict(self, img):
        logits = self.model(img).logits
        predicted_label = logits.argmax(-1).item()
        id = self.id2labels[predicted_label]
        confidence = logits.softmax(-1).max().item()
        return ReIdResult(id=id, confidence=confidence)

    def __call__(self, *args, **kwds):
        return self.predict(*args, **kwds)