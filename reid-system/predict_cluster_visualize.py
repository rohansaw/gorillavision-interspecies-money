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
from sklearn_som.som import SOM


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


def visualize_embeddings_som(embeddings, labels, k, db_embeddings=None):
    embeddings_som = SOM(m=10, n=10, dim=256)
    embeddings_som.fit(embeddings)
    predictions = embeddings_som.predict(embeddings)
    plt.figure(figsize=(10, 10))
    plt.title("SOM Clustering")

    # todo how to visualize the embeddings in 2D


def visualize_embeddings_unsupervised(
    embeddings, file_names, k, out_path, db_embeddings=None
):
    # visualize the embeddings in pca and perform unsupervised clustering.

    pca = PCA(n_components=2)

    # if we have db_embeddings use them and give them a red border, but also include them in the clustering
    if db_embeddings is not None:
        combined_embeddings = np.vstack([embeddings, db_embeddings])
        reduced_embeddings = pca.fit_transform(combined_embeddings)

        # Split the transformed embeddings back
        reduced_main_embeddings = reduced_embeddings[: len(embeddings)]
        reduced_db_embeddings = reduced_embeddings[len(embeddings) :]

        kmeans = KMeans(n_clusters=k)
        kmeans.fit(combined_embeddings)
        plt.scatter(
            reduced_db_embeddings[:, 0],
            reduced_db_embeddings[:, 1],
            color="blue",
            label="Database",
            edgecolors="red",
            facecolors="none",
        )
        plt.scatter(
            reduced_main_embeddings[:, 0],
            reduced_main_embeddings[:, 1],
            c=kmeans.labels_[: len(embeddings)],
            label="New",
        )
    else:
        reduced_embeddings = pca.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(embeddings)
        plt.scatter(
            reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=kmeans.labels_
        )

    plt.title("Embeddings Visualization with Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    # create a folder for each cluster and save the images in it
    out_path_pca_clusters = os.path.join(out_path, "pca_clusters")
    for i in range(k):
        if not os.path.exists(os.path.join(out_path_pca_clusters, str(i))):
            os.makedirs(os.path.join(out_path_pca_clusters, str(i)))
    for i, label in enumerate(kmeans.labels_):
        shutil.copy(
            os.path.join(args.images_path, file_names[i] + ".png"),
            os.path.join(out_path_pca_clusters, str(label), file_names[i] + ".png"),
        )

    # create kmeans wirthout pca for comparison
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)

    # save the clustered images
    out_path_no_pca_clusters = os.path.join(out_path, "no_pca_clusters")
    for i in range(k):
        if not os.path.exists(
            os.path.join(out_path_no_pca_clusters, str(i) + "_kmeans")
        ):
            os.makedirs(os.path.join(out_path_no_pca_clusters, str(i) + "_kmeans"))
    for i, label in enumerate(kmeans.labels_):
        shutil.copy(
            os.path.join(args.images_path, file_names[i] + ".png"),
            os.path.join(
                out_path_no_pca_clusters, str(label) + "_kmeans", file_names[i] + ".png"
            ),
        )


def visualize_embeddings(embeddings, labels, db_embeddings=None, db_labels=None):
    pca = PCA(n_components=2)
    plt.figure(figsize=(8, 8))

    if db_embeddings is not None:
        combined_embeddings = np.vstack([embeddings, db_embeddings])
        reduced_embeddings = pca.fit_transform(combined_embeddings)

        # Split the transformed embeddings back
        reduced_main_embeddings = reduced_embeddings[: len(embeddings)]
        reduced_db_embeddings = reduced_embeddings[len(embeddings) :]

        plt.scatter(
            reduced_db_embeddings[:, 0],
            reduced_db_embeddings[:, 1],
            color="blue",
            label="Database",
        )
        plt.scatter(
            reduced_main_embeddings[:, 0],
            reduced_main_embeddings[:, 1],
            color="red",
            label="New",
        )
    else:
        reduced_embeddings = pca.fit_transform(embeddings)
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

    parser.add_argument(
        "--num_individuals",
        type=int,
        required=False,
        help="Number of distinct individuals. Required for unsupervised clustering.",
    )

    parser.add_argument(
        "--save_embeddings",
        type=bool,
        required=False,
        default=False,
        help="Save the embeddings to a file.",
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

    if args.save_embeddings:
        np.save(os.path.join(args.out_path, "embeddings.npy"), embeddings)
        np.save(os.path.join(args.out_path, "file_names.npy"), file_names)

    # save the images with their predicting labels, and combine all images with the same id in one folder
    for image_path, id in zip(file_names, ids):

        if not os.path.exists(os.path.join(args.out_path, str(id))):
            os.makedirs(os.path.join(args.out_path, str(id)))
        shutil.copy(
            os.path.join(args.images_path, image_path + ".png"),
            os.path.join(args.out_path, str(id), image_path + ".png"),
        )

    # visualize_embeddings_som(embeddings, ids, args.num_individuals, db_embeddings)

    if args.db_path:
        visualize_embeddings(embeddings, ids, db_embeddings, db_labels)
    else:
        visualize_embeddings(embeddings, file_names)

    if args.num_individuals:
        out_path = args.out_path
        if args.db_path:
            visualize_embeddings_unsupervised(
                embeddings, file_names, args.num_individuals, out_path, db_embeddings
            )
        else:
            visualize_embeddings_unsupervised(
                embeddings, file_names, args.num_individuals, out_path
            )
