import argparse
import os
import csv
from cv2 import imread
from gorillavision.model.triplet import TripletLoss
import torch
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from gorillavision.utils.image import transform_image
import wandb


def create_db(
    image_folder,
    model,
    input_width,
    input_height,
    img_preprocess,
    return_embedding_images=False,
):
    labels = []
    embeddings = []
    dimensions = []

    all_data = []
    for folder in os.listdir(image_folder):
        if folder == "script.sh":
            continue
        for img_file in tqdm(os.listdir(os.path.join(image_folder, folder))):
            _, ext = os.path.splitext(img_file)
            if ext not in [".png", ".jpg", ".jpeg"]:
                continue
            with torch.no_grad():
                img_path = os.path.join(image_folder, folder, img_file)
                img = transform_image(
                    imread(img_path), (input_width, input_height), img_preprocess
                )
                labels.append(folder)
                embedding = model(img).numpy()[0]
                embeddings.append(embedding)
                if return_embedding_images:
                    all_data.append([folder, wandb.Image(img), *embedding])
    for idx, _ in enumerate(embeddings[0]):
        dimensions.append(f"dim_{idx}")
    embeddings_data = pd.DataFrame(
        data=all_data, columns=["target", "image", *dimensions]
    )

    return np.array(labels), np.array(embeddings), embeddings_data


def create_db_interspecies_format(
    image_folder,
    model,
    input_width,
    input_height,
    img_preprocess,
    return_embedding_images=False,
):
    labels = []
    embeddings = []

    for individual_folder in os.listdir(image_folder):
        for img_path in os.listdir(os.path.join(image_folder, individual_folder)):
            _, ext = os.path.splitext(img_path)
            if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            label = individual_folder
            with torch.no_grad():
                full_img_path = os.path.join(image_folder, individual_folder, img_path)
                img = imread(full_img_path)
                img = transform_image(img, (input_width, input_height), img_preprocess)
                labels.append(label)
                embedding = model(img).numpy()[0]
                embeddings.append(embedding)

    return np.array(labels), np.array(embeddings)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-c",
        "--conf",
        help="path to configuration file in config folder",
        default="configs/config.json",
    )
    argparser.add_argument(
        "-f",
        "--format",
        help='use "im" for interspecies-money groundtruth',
        default=None,
    )
    args = argparser.parse_args()
    with open(args.conf) as config_buffer:
        config = json.loads(config_buffer.read())
    image_folder = config["create_db"]["image_folder"]
    model = TripletLoss.load_from_checkpoint(config["create_db"]["model_path"])
    input_width = config["model"]["input_width"]
    input_height = config["model"]["input_height"]
    img_preprocess = config["model"]["img_preprocess"]
    if not args.format:
        labels, embeddings, _ = create_db(
            image_folder, model, input_width, input_height, img_preprocess
        )
    elif args.format == "im":
        labels, embeddings = create_db_interspecies_format(
            image_folder, model, input_width, input_height, img_preprocess
        )
    else:
        print("Unsupported Format.")

    if not os.path.exists(config["create_db"]["db_path"]):
        os.mkdir(config["create_db"]["db_path"])
    np.save(
        os.path.join(config["create_db"]["db_path"], "labels.npy"), np.array(labels)
    )
    np.save(
        os.path.join(config["create_db"]["db_path"], "embeddings.npy"),
        np.array(embeddings),
    )
