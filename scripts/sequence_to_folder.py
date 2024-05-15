import os
import shutil


def read_label(file_path, id):
    with open(file_path) as label_file:
        for line in label_file:
            if line.split(" ")[0] == id:
                corners = line.split()[1:]
                return [float(val) for val in corners]
    return None


def yolobbox2bbox(x, y, w, h, img_w, img_h):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h


label_id = "0"  # describes which label should be read. Currently there must be only one label of this type
images_folder = (
    "/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/all_images"
)
lables_folder = "/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/kwitonda_manually_labeled/obj_train_data"
output_folder = os.path.join("/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda", "sequences")

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


file_path = "/Users/lukaslaskowski/Documents/HPI/gorillavision/image_sequences.txt"


num_cropped = 0
with open(file_path, 'r') as txt_file:
        for idx, line in enumerate(txt_file):
            elements = line.strip().split(',')
            os.makedirs(os.path.join(output_folder, str(idx)),exist_ok=True)
            for file_name in elements:
                file_extension = "jpg"
                #file_name, file_extension = os.path.splitext(file)
                label_path = os.path.join(lables_folder, f"{file_name[:-4].strip()}.txt")
                print(label_path)
                if not os.path.exists(label_path):
                    continue
                bbox = read_label(label_path, label_id)
                if not bbox:
                    continue
                shutil.copy(os.path.join(images_folder, file_name.strip()), os.path.join(output_folder, str(idx), file_name[:-4].strip() + ".png"))
                #new_path = os.path.join(output_folder, file_name + ".png")
                num_cropped += 1

print(f"Files in Images folder: {len(os.listdir(images_folder))}")
print(f"Cropped {num_cropped} images")
