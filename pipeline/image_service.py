import os
from pathlib import Path
import shutil
import logging
import cv2


class ImageService:
    def __init__(
        self,
        input_dataset_path,
        output_root,
        allowed_img_extensions,
        allowed_video_extensions,
        save_output_videos,
    ):
        self.input_dataset_path = Path(input_dataset_path)
        self.output_root = Path(output_root)
        self.save_output_videos = save_output_videos
        self.validate_data(allowed_img_extensions, allowed_video_extensions)
        self.copy_original_files()
        self.classes = [d for d in self.input_dataset_path.iterdir() if d.is_dir()]
        self.class_iters = {
            cls: iter(
                [
                    img
                    for img in cls.iterdir()
                    if img.suffix.lower().endswith(
                        tuple(allowed_img_extensions + allowed_video_extensions)
                    )
                ]
            )
            for cls in self.classes
        }
        self.current_class = None
        self.current_image = None

    def validate_data(self, allowed_img_extensions, allowed_video_extensions):
        print(self.input_dataset_path)
        sub_folder = os.listdir(self.input_dataset_path)
        if not sub_folder:
            logging.error("The dataset folder does not contain any subfolders.")
            raise FileNotFoundError
        for folder in sub_folder:
            files = os.listdir(os.path.join(self.input_dataset_path, folder))
            if not any(
                file.lower().endswith(
                    tuple(allowed_img_extensions + allowed_video_extensions)
                )
                for file in files
            ):
                logging.error(f"The subfolder {folder} does not contain any images.")
                raise FileNotFoundError
        os.makedirs(self.output_root, exist_ok=True)

    def copy_original_files(self):
        output_original_files = os.path.join(self.output_root, "original_files")
        os.makedirs(output_original_files, exist_ok=True)
        for folder in os.listdir(self.input_dataset_path):
            os.makedirs(os.path.join(output_original_files, folder), exist_ok=True)
            for file in os.listdir(os.path.join(self.input_dataset_path, folder)):
                if self.image_already_processed(Path(file), folder):
                    logging.info(
                        f"File {file} in class {folder} already processed. Skipping."
                    )
                    continue
                shutil.copy(
                    os.path.join(self.input_dataset_path, folder, file),
                    os.path.join(output_original_files, folder, file),
                )

    def __iter__(self):
        return self

    def __next__(self):
        for cls, it in self.class_iters.items():
            try:
                self.current_image = next(it)
                self.current_class = cls.name
                return self.current_image, self.current_class
            except StopIteration:
                continue
        raise StopIteration

    def image_already_processed(self, image, folder):
        my_file = Path("/path/to/file")
        cropped_path = os.path.join(self.output_root, "cropped", folder)
        original_path = os.path.join(self.output_root, "original_files", folder)
        label_path = os.path.join(self.output_root, "labels", folder)

        def check_if_file_in_path(file_path):
            return (
                len(
                    list(filter(lambda name: image.stem in name, os.listdir(file_path)))
                )
                > 0
                if os.path.exists(file_path)
                else False
            )

        return check_if_file_in_path(original_path) and check_if_file_in_path(
            label_path
        )

    def save_yolo_output(self, output):
        if self.current_class is None or self.current_image is None:
            raise RuntimeError("No image currently being processed")
        output_dir = self.output_root / self.current_class
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.current_image.stem}_output.txt"
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Output saved to {output_path}")
