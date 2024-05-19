
import torch
from PIL import Image

import torchvision.transforms as T

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").eval()

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = "cpu"

dinov2_vits14.to(device)

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img_path: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img_path)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img
    
    
import glob
from tqdm import tqdm


for id in range(0, 1):
    results = []

    images = glob.glob(f'/data/sequences_cropped/{id}/*.png')
    for img_path in tqdm(images):
        embeddings = dinov2_vits14(load_image(img_path).to(device))
        embedding = embeddings[0].detach().cpu().numpy()
        results.append(embedding)

    import numpy as np
    np.save(f"/workspace/analysis/results_dino/{id}", np.array(results))
# id = 34
# results = []

# images = glob.glob(f'/data/sequences_cropped/{id}/*.png')
# for img_path in tqdm(images):
#     embeddings = dinov2_vits14(load_image(img_path).to(device))
#     embedding = embeddings[0].detach().cpu().numpy()
#     results.append(embedding)
    
# import numpy as np
# np.save(f"{id}", np.array(results))



