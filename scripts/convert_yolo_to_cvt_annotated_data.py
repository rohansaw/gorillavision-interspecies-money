# Zip the outcoming folder and upload it to cvt to import preannotated images
import os
import shutil

image_folder = "/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/Feb2024"  # User needs to replace this with the actual path
label_folder = "/Users/lukaslaskowski/Documents/HPI/gorillavision/face_detection/yolov7/runs/detect/kwitonda_nose_prints_230324_imgsize_224_conf_0_03/labels"  # User needs to replace this with the actual path
root_path = "./test_folder"  # This is where the user wants to create the obj_train_data folder and other files
train_data_folder = os.path.join(root_path, "obj_train_data")
os.makedirs(train_data_folder, exist_ok=True)

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        try:
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            shutil.copy(os.path.join(label_folder, txt_filename), train_data_folder)
            shutil.copy(os.path.join(image_folder, filename), train_data_folder)
        except:
            print(f"{txt_filename} does not exist")

with open(os.path.join(root_path, "obj.data"), 'w') as f:
    f.write("classes = 1\n")
    f.write("names = data/obj.names\n")
    f.write("train = data/train.txt\n")

with open(os.path.join(root_path, "obj.names"), 'w') as f:
    f.write("gorilla\n")

with open(os.path.join(root_path, "train.txt"), 'w') as f:
    for filename in os.listdir(train_data_folder):
        if filename.endswith(".jpg"):
            f.write(f"data/obj_train_data/{filename}\n")
