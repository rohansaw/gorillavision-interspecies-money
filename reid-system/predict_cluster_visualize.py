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

    def transform_image(self, image):
        return self.transform(image)

    def predict_embedding(self, img_path):
        img = cv2.imread(str(img_path))
        img = transform_image(img, self.img_sz, "crop")
        with torch.no_grad():
            embedding = self.identity_model(img).numpy()
        return embedding


def visualize_embeddings(embeddings, labels):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    for i, label in enumerate(labels):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=label)
    plt.legend()
    plt.title("Embeddings Visualization")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


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
        "--db_path",
        type=str,
        required=False,
        help="Path to the folder containing the database files.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the folder where the outputs will be saved.",
    )

    args = parser.parse_args()

    embedder = ImageEmbedder(args.model_path, (224, 224))
    embeddings = []
    file_names = []
    ids = []

    if args.db_path:
        db_embeddings = np.load(os.path.join(args.db_path, "embeddings.npy"))
        db_labels = np.load(os.path.join(args.db_path, "labels.npy"))
        knn_classifier = neighbors.KNeighborsClassifier()
        knn_classifier.fit(db_embeddings, db_labels)

    for image_path in Path(args.images_path).glob("*.png"):
        embedding = embedder.predict_embedding(image_path)
        embeddings.append(embedding[0])
        file_names.append(image_path.stem)

        if args.db_path:
            predicted_id = knn_classifier.predict(embedding)
            ids.append(predicted_id[0])

    embeddings = np.array(embeddings)

    # save the images with their predicting labels, and combine all images with the same id in one folder
    for image_path, id in zip(file_names, ids):

        if not os.path.exists(os.path.join(args.out_path, str(id))):
            os.makedirs(os.path.join(args.out_path, str(id)))
        shutil.copy(
            os.path.join(args.images_path, image_path + ".png"),
            os.path.join(args.out_path, str(id), image_path + ".png"),
        )

    visualize_embeddings(embeddings, ids)
