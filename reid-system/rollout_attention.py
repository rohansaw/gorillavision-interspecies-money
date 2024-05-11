import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from gorillavision.model.triplet import TripletLoss
from gorillavision.utils.image import transform_image

model_path = "/home/rohan/Documents/projects/interspecies_money/data/models/identification_model.ckpt"
img_path = "/home/rohan/Documents/projects/interspecies_money/data/out/IKR/IMG_2340.png"
img_sz = (224, 224)

# Load the model and switch to eval mode
model = TripletLoss.load_from_checkpoint(model_path)
model.eval()


def attention_rollout(attentions, head_fusion_method="mean"):
    rollout = torch.eye(attentions[0].size(-1))
    print(attentions[0].size(-1))
    print("attentions shapes: ", [attention.size() for attention in attentions])

    for attention in attentions[1:]:
        attention_heads = attention.squeeze(0)
        print("attention_heads shape: ", attention_heads.size())
        if head_fusion_method == "mean":
            attention_heads_fused = attention_heads.mean(0)
        else:
            attention_heads_fused = attention_heads.min(0).values

        print("attention_heads_fused shape: ", attention_heads_fused.size())
        rollout = torch.matmul(rollout, attention_heads_fused)

    # rollout /= rollout.max()
    print(rollout.shape)
    return rollout


def visualize_attention(image_path, attention_map):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(img_sz)  # Resize to ensure size consistency
    attention_map_resized = cv2.resize(
        attention_map, img_sz, interpolation=cv2.INTER_CUBIC
    )

    # Apply Gaussian blur to the attention map to smooth out the grid effect
    attention_map_smoothed = cv2.GaussianBlur(attention_map_resized, (5, 5), sigmaX=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img, alpha=0.8)
    plt.imshow(
        attention_map_smoothed, cmap="jet", alpha=0.5
    )  # Overlay the smoothed attention map
    plt.title("Attention Map")
    plt.axis("off")
    plt.show()


# Read and preprocess the image
img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = transform_image(img, img_sz, "crop")
embeddings, attentions = model(img)

# Generate and visualize the attention rollout
rollout_attention = attention_rollout(attentions).detach().numpy()
visualize_attention(img_path, rollout_attention)
