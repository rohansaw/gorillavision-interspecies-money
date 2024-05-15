from sklearn.cluster import KMeans
import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from torchvision import transforms
from sklearn import neighbors
from sklearn.decomposition import PCA
from pathlib import Path
from PIL import Image
from gorillavision.utils.image import transform_image
from gorillavision.model.triplet import TripletLoss
import cv2
import os
import shutil


class ImageEmbedder:
    def __init__(self, model_path, img_sz):
        self.identity_model = TripletLoss.load_from_checkpoint(model_path)
        self.identity_model.eval()
        self.img_sz = img_sz

    def predict_embedding(self, img_path):
        img = cv2.imread(img_path)
        img = transform_image(img, self.img_sz, "crop")
        with torch.no_grad():
            embedding = self.identity_model(img).numpy()
        return embedding


def create_embeddings_by_individual_folder(model_path, images_path):
    embedder = ImageEmbedder(model_path, (224, 224))
    embeddings = []
    file_names = []
    ids = []

    for individual_folder in os.listdir(images_path):
        if not os.path.isdir(os.path.join(images_path, individual_folder)):
            continue
        for image_path in os.listdir(os.path.join(images_path, individual_folder)):
            img_name, ext = os.path.splitext(image_path)
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                print(image_path)
                print(ext)
                continue
            full_image_path = os.path.join(images_path, individual_folder, image_path)
            embedding = embedder.predict_embedding(full_image_path)
            embeddings.append(embedding[0])
            image_path = os.path.join(individual_folder, image_path)
            file_names.append(image_path)
            ids.append(individual_folder.upper())

    embeddings = np.array(embeddings)
    file_names = np.array(file_names)
    ids = np.array(ids)
    return embeddings, file_names, ids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file."
    )
    parser.add_argument(
        "--images_path",
        type=str,
        required=True,
        help="Path to the folder containing images.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=False,
        default=".",
        help="Path to the folder where the outputs will be saved.",
    )

    args = parser.parse_args()

    embeddings, file_names, ids = create_embeddings_by_individual_folder(
        args.model_path, args.images_path
    )

    np.save(os.path.join(args.out_path, "embeddings.npy"), embeddings)
    np.save(os.path.join(args.out_path, "file_names.npy"), file_names)
    np.save(os.path.join(args.out_path, "individual_ids.npy"), ids)
