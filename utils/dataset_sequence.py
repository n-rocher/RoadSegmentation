import os
import cv2
import random
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
from glob import glob

CAMVID_CATEGORIES = {
    1: {"name": 'sky', "color": [[128, 128, 128]]},
    2: {"name": 'building', "color": [[128, 0, 0]]},
    3: {"name": 'column_pole', "color": [[192, 192, 128]]},
    4: {"name": 'road', "color": [[128, 64, 128]]},
    5: {"name": 'sidewalk', "color": [[0, 0, 192]]},
    6: {"name": 'tree', "color": [[128, 128, 0]]},
    7: {"name": 'sing_symbol', "color": [[192, 128, 128]]},
    8: {"name": 'fence', "color": [[64, 64, 128]]},
    9: {"name": 'car', "color": [[64, 0, 128]]},
    10: {"name": 'pedestrian', "color": [[64, 64, 0]]},
    11: {"name": 'bicyclist', "color": [[0, 128, 192]]}
}

CITYSCAPE_CATEGORIES = {
    0: {"name": 'Background', "id": [0, 1, 2, 3, 4, 5, 14, 15, 16], "color": [0, 0, 0]},
    1: {"name": 'Ground', "id": [6], "color": [81, 0, 81]},
    2: {"name": 'Road', "id": [7], "color": [128, 64, 128]},
    3: {"name": 'Sidewalk', "id": [8], "color": [244, 35, 232]},
    4: {"name": 'Parking', "id": [9], "color": [250, 170, 160]},
    5: {"name": 'Rail track', "id": [10], "color": [230, 150, 140]},
    6: {"name": 'Building', "id": [11], "color": [70, 70, 70]},
    7: {"name": 'Wall', "id": [12], "color": [102, 102, 156]},
    8: {"name": 'Fence', "id": [13], "color": [190, 153, 153]},
    9: {"name": 'Pole', "id": [17, 18], "color": [153, 153, 153]},
    10: {"name": 'Traffic light', "id": [19], "color": [250, 170, 30]},
    11: {"name": 'Traffic sign', "id": [20], "color": [220, 220, 0]},
    12: {"name": 'Vegetation', "id": [21], "color": [107, 142, 35]},
    13: {"name": 'Terrain', "id": [22], "color": [152, 251, 152]},
    14: {"name": 'Sky', "id": [23], "color": [70, 130, 180]},
    15: {"name": 'Person', "id": [24, 25], "color": [220, 20, 60]},
    16: {"name": 'Car', "id": [26], "color": [0, 0, 142]},
    17: {"name": 'Truck', "id": [27], "color": [0, 0, 70]},
    18: {"name": 'Bus', "id": [28], "color": [0, 60, 100]},
    19: {"name": 'Caravan', "id": [29], "color": [0, 0, 90]},
    20: {"name": 'Trailer', "id": [30], "color": [0, 0, 110]},
    21: {"name": 'Train', "id": [31], "color": [0, 80, 100]},
    22: {"name": 'Motorcycle', "id": [32], "color": [0, 0, 230]},
    23: {"name": 'Bicycle', "id": [33], "color": [119, 11, 32]}
}

# FIXME: Not working
class CamvidSequenceDataset(keras.utils.Sequence):

    def __init__(self, batch_size, sequence_size, img_size, dataset_type):

        self.dataset_folder = r"F:\\CAMVID\\"
        self.batch_size = batch_size
        self.sequence_size = sequence_size
        self.img_size = img_size

        labels = self.getLabels()

        self.getSequence(labels)

    def getLabels(self):
        labels = glob(os.path.join(self.dataset_folder, "labels", "*.png"))
        return labels

    def getSequence(self, labels):
        images = glob(os.path.join(self.dataset_folder, "images", "*.png"))
        images_name = ["_".join(os.path.basename(n).split(".")[:-1]) for n in glob(os.path.join(self.dataset_folder, "images", "*.png"))]
        print(images_name)
        for label in labels:
            a = os.path.basename(label)
            a = a.split("_")

            file_id = 0
            if "f" in a[1]:
                file_id = int(a[1][1:])
            else:
                file_id = int(a[1])

            print("_".join(a[:-1]) in images_name)

    def classes(self):
        return len(self.CAMVID_CATEGORIES) + 1

    def labels(self):
        l = {0: {"name": "Background", "color": [0, 0, 0]}}
        for i in self.CAMVID_CATEGORIES:
            l[i] = self.CAMVID_CATEGORIES[i]["name"]
        return l

    def colors(self):
        l = {0: {"name": "Background", "color": [0, 0, 0]}}
        for i in self.CAMVID_CATEGORIES:
            l[i] = {"name": self.CAMVID_CATEGORIES[i]["name"], "color": self.CAMVID_CATEGORIES[i]["color"][0]}
        return l

    def name(self):
        return "CamvidSequenceDataset"

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + (3,) + self.img_size, dtype="float32")
        past_image = np.zeros((self.batch_size,) + (self.sequence_size,) + (3,) + self.img_size, dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            frame = cv2.cvtColor(np.array(load_img(path, target_size=self.img_size)), cv2.COLOR_BGR2RGB)
            x[j] = frame / 255.

        # Chargement du masque et traitement
        ins_255 = np.ones(self.img_size) * 255
        truth_x = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size, dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):
            mask = np.array(load_img(path, target_size=self.img_size, color_mode="rgb"))  # On charge le masque

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for i, id_category in enumerate(self.CATEGORIES, start=1):

                data_category = self.CATEGORIES[id_category]

                for color in data_category["color"]:

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, tuple(color), tuple(color))

                    truth_x[j, i] = truth_x[j, i] + (test >= 1)  # Permet d'avoir que des 1 ou 0

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + res

            truth_x[j, 0] = instance == 0

        truth_x = np.moveaxis(truth_x, 1, -1)

        return [x, past_image], truth_x


class CityscapeSequenceDataset(keras.utils.Sequence):

    def __init__(self, batch_size, sequence_length, sequence_delay, img_size, dataset_type):

        self.dataset_folder = r"C:\\Cityscapes\\" if dataset_type == "train" else r"F:\\Cityscapes\\"
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.sequence_delay = sequence_delay
        self.sequences = self.getSequences()

        self.NORMALIZATION_MEAN = [123.675, 116.28, 103.53]
        self.NORMALIZATION_STD = [58.395, 57.12, 57.375]

        # self.moveFiles()

        self.CATEGORIES = CITYSCAPE_CATEGORIES

    def getSequences(self):
        sequence_list = []

        def processFileName(item):
            item = os.path.basename(item).split("_")
            item[1] = int(item[1])
            item[2] = int(item[2])
            return item

        for city in os.listdir(self.dataset_folder + "gtFine/" + self.dataset_type):

            folder = "/" + self.dataset_type + "/" + city + "/"

            image = glob(self.dataset_folder + "leftImg8bitSequence" + folder + "/*_leftImg8bit.png")
            labels = glob(self.dataset_folder + "gtFine" + folder + "/*_labelIds.png")

            image = list(map(lambda s: os.path.basename(s), image))
            labels_processed = list(map(processFileName, labels))

            item = processFileName(image[0])

            for index_label, (city, sequence, index, _, _) in enumerate(labels_processed):
                img_sequence = []
                doable = True

                for i in range(0, self.sequence_length * (self.sequence_delay + 1), self.sequence_delay):
                    looking_for = '_'.join([city, str(sequence).zfill(6), str(index - i).zfill(6), item[-1]])
                    doable = image.index(looking_for) >= 0
                    if doable is not True:
                        break
                    img_sequence.append(self.dataset_folder + "leftImg8bitSequence" + folder + looking_for)

                if doable is True:
                    sequence_list.append({
                        "img_sequence": img_sequence,
                        "label": labels[index_label],
                        "flip": True
                    })

                    sequence_list.append({
                        "img_sequence": img_sequence,
                        "label": labels[index_label],
                        "flip": False
                    })

        random.shuffle(sequence_list)
        return sequence_list

    def moveFiles(self):

        from tqdm import tqdm
        import shutil
        import os

        for i in tqdm(range(len(self.sequences))):

            images, mask = self.sequences[i]

            image_0_folder = os.path.dirname("C" + images[0][1:])
            mask_folder = os.path.dirname("C" + mask[1:])

            if not os.path.exists(image_0_folder):
                os.makedirs(image_0_folder)

            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)

            for image in images:
                shutil.copy(image, "C" + image[1:])

            shutil.copy(mask, "C" + mask[1:])

    def classes(self):
        return len(self.CATEGORIES)

    def labels(self):
        output = {}
        for i in self.CATEGORIES:
            output[i] = self.CATEGORIES[i]["name"]
        return output

    def colors(self):
        return self.CATEGORIES

    def name(self):
        return "CityscapeSequenceDataset-Length-" + str(self.sequence_length) + "-Delay-" + str(self.sequence_delay)

    def __len__(self):
        return len(self.sequences) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_paths = self.sequences[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + (3,) + self.img_size[::-1], dtype="float32")
        old_x = np.zeros((self.batch_size,) + (self.sequence_length,) + (3,) + self.img_size[::-1], dtype="float32")
        y = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size[::-1], dtype="uint8")

        # Chargement des batchs
        for j, batch_data in enumerate(batch_paths):

            assert len(batch_data["img_sequence"]) == self.sequence_length + 1, "ATTENTION : BATCH=" + str(idx) + " -- len(images_path)=" + str(len(images_path)) + " != self.sequence_length = " + str(self.sequence_length)

            # Loading current image
            x_image = cv2.imread(batch_data["img_sequence"][0])
            x_image = cv2.resize(x_image, self.img_size)
            x_image = cv2.cvtColor(x_image, cv2.COLOR_BGR2RGB)

            # Flipping horizontaly if True
            if batch_data["flip"] is True:
                x_image = cv2.flip(x_image, 1)

            # Normalization
            x_image = (x_image - self.NORMALIZATION_MEAN)  / self.NORMALIZATION_STD

            x[j] = np.rollaxis(x_image, 2, 0)

            # Loading sequence images
            for i in range(0, self.sequence_length):
                
                # Reading image
                old_x_i = cv2.imread(batch_data["img_sequence"][i+1])
                old_x_i = cv2.resize(old_x_i, self.img_size)
                old_x_i = cv2.cvtColor(old_x_i, cv2.COLOR_BGR2RGB)

                # Flipping horizontaly if True
                if batch_data["flip"] is True:
                    old_x_i = cv2.flip(old_x_i, 1)

                # Normalization
                old_x_i = (old_x_i - self.NORMALIZATION_MEAN) / self.NORMALIZATION_STD

                old_x[j, i] = np.rollaxis(old_x_i, 2, 0)

            # Chargement du masque de l'image courante
            label_image = cv2.imread(batch_data["label"], cv2.IMREAD_GRAYSCALE)
            label_image = cv2.resize(label_image, self.img_size)

            if batch_data["flip"] is True:
                label_image = cv2.flip(label_image, 1)

            # For every categories in the list
            for id_category in self.CATEGORIES:

                data_category = self.CATEGORIES[id_category]

                for id in data_category["id"]:

                    # We select pixels belonging to that category
                    test = label_image == id
                    y[j, id_category] = y[j, id_category] + test

        y[y > 0] = 1
        y = np.array(y, dtype=np.uint8)

        return [x, old_x], y


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import tensorflow as tf

    IMG_SIZE = (384, 384)
    BATCH_SIZE = 2
    SEQUENCE_LENGTH = 2
    SEQUENCE_DELAY = 2

    train_gen = CityscapeSequenceDataset(BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_DELAY, IMG_SIZE, "train")
    print("    Training :", len(train_gen), "batchs")

    filename = train_gen.name()

    for id in range(len(train_gen)):
        print("BATCH-", id)
        data = train_gen.__getitem__(id)

        for img_i, old_image_i, mask_i in zip(data[0][0], data[0][1], data[1]):

            fig, axs = plt.subplots(1, 2 + old_image_i.shape[0])

            mask_i = tf.argmax(mask_i, axis=0)
            mask_i = mask_i[..., tf.newaxis]

            axs[0].imshow(np.rollaxis(img_i, 0, 3))

            for i in range(0, old_image_i.shape[0]):
                axs[1 + i].imshow(np.rollaxis(old_image_i[i], 0, 3))

            axs[-1].imshow(mask_i)

            fig.show()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
