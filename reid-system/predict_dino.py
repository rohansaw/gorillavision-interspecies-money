
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14").eval()

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device = "cuda:0"
dinov2_vits14.to(device)
transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

def load_image(img_path: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img_path)
    transformed_img = transform_image(img)[:3].unsqueeze(0)
    return transformed_img


path='/workspace/data/crops_and_embeddings/cameratrap_02_2024/'
file_names=np.load(f'{path}/embeddings/file_names.npy')


images = [f"{path}cropped_faces/{img}" for img in file_names]
results = []
for img_path in tqdm(images):
    embeddings = dinov2_vits14(load_image(img_path + '.jpg').to(device))
    embedding = embeddings[0].detach().cpu().numpy()
    results.append(embedding)


np.save(f"{path}/embeddings/embeddings_dino.npy", np.array(results))




