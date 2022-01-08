import os
import cv2
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

def getImagesAndMasksPath(images_path, masks_path):

    list_masks = os.listdir(masks_path)

    list_ids = [os.path.splitext(mask_name)[0] for mask_name in list_masks]

    input_train_img_paths = [os.path.join(images_path, id + ".jpg") for id in list_ids]
    target_train_img_paths = [os.path.join(masks_path, id + ".png") for id in list_ids]

    return input_train_img_paths, target_train_img_paths

class MapillaryVistasDataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, dataset_type):

        dataset_type = "training" if dataset_type == "train" else ("validation" if dataset_type == "val" else dataset_type)

        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths, self.target_img_paths = getImagesAndMasksPath("F:\\Mapillary Vistas\\" + dataset_type + "\\images\\", "F:\\Mapillary Vistas\\" + dataset_type + "\\v1.2\labels\\")

        self.CATEGORIES = {
            1: {"name": "Road", "color": [[128, 64, 128]]},
            2: {"name": "Lane", "color": [[255, 255, 255]]},
            3: {"name": "Crosswalk", "color": [[140, 140, 200], [200, 128, 128]]},
            4: {"name": "Curb", "color": [[196, 196, 196]]},
            5: {"name": "Sidewalk", "color": [[244, 35, 232]]},

            6: {"name": "Traffic Light", "color": [[250, 170, 30]]},
            7: {"name": "Traffic Sign", "color": [[220, 220, 0]]},

            8: {"name": "Person", "color": [[220, 20, 60]]},
            9: {"name": "Bicyclist", "color": [[255, 0, 0]]},
            10: {"name": "Motorcyclist", "color": [[255, 0, 100]]},

            11: {"name": "Bicycle", "color": [[119, 11, 32]]},
            12: {"name": "Bus", "color": [[0, 60, 100]]},
            13: {"name": "Car", "color": [[0, 0, 142]]},
            14: {"name": "Motorcycle", "color": [[0, 0, 230]]},
            15: {"name": "Truck", "color": [[0, 0, 70]]}
        }

    def classes(self):
        return len(self.CATEGORIES) + 1

    def name(self):
        return "MapillaryVistasDataset"

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            frame = np.array(load_img(path, target_size=self.img_size))
            x[j] = frame / 255.

        # Chargement du masque et traitement
        ins_255 = np.ones(self.img_size) * 255
        y = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size, dtype="uint8")

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

                    y[j, i] = y[j, i] + (test >= 1)  # Permet d'avoir que des 1 ou 0

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + res

            y[j, 0] = instance == 0

        y = np.moveaxis(y, 1, -1)

        return x, y

class A2D2Dataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size):

        self.dataset_folder = r"F:\\A2D2 Camera Semantic\\"
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths, self.target_img_paths = self.getData()

        self.CATEGORIES = {
            1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
            2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
            3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
            4: {"name": "Curb", "color": [[128, 128, 0]]},
            5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

            6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
            7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

            8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},
            9: {"name": "Bicyclist", "color": []},
            10: {"name": "Motorcyclist", "color": []},

            11: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
            12: {"name": "Bus", "color": []},
            13: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
            14: {"name": "Motorcycle", "color": []},
            15: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0]]}
        }

    def classes(self):
        return len(self.CATEGORIES) + 1

    def name(self):
        return "A2D2Dataset"

    def getData(self):
        '''
        Permet de trouver le nom des fichiers du jeux de donnée A2D2
        '''
        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(self.dataset_folder, item) for item in os.listdir(self.dataset_folder) if os.path.isdir(self.dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return data_image, data_label

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = np.array(load_img(path, target_size=self.img_size)) / 255.

        # Chargement du masque et traitement
        ins_255 = np.ones(self.img_size) * 255
        y = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = np.array(load_img(path, target_size=self.img_size, color_mode="rgb"))

            cv2.imshow("mask", mask)

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for i, id_category in enumerate(self.CATEGORIES, start=1):

                data_category = self.CATEGORIES[id_category]

                for color in data_category["color"]:

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, tuple(color), tuple(color))

                    y[j, i] = y[j, i] + (test >= 1)  # Permet d'avoir que des 1 ou 0

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + res

            cv2.imshow("instance", instance * 255)
            y[j, 0] = instance == 0

        y = np.moveaxis(y, 1, -1)

        return x, y

class MultiDataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, dataset_type):

        dataset_type = "training" if dataset_type == "train" else ("validation" if dataset_type == "val" else dataset_type)

        self.batch_size = batch_size
        self.img_size = img_size

        # Chargement des données
        self.vistas_dataset = self.getVistasData(dataset_type)
        self.a2d2_dataset = self.getA2D2Data(dataset_type)
        self.dataset = self.vistas_dataset + self.a2d2_dataset
        random.shuffle(self.dataset)

        self.CATEGORIES = {
            "VISTAS": {
                1: {"name": "Road", "color": [[128, 64, 128], [110, 110, 110]]},
                2: {"name": "Lane", "color": [[255, 255, 255], [250, 170, 29], [250, 170, 28], [250, 170, 26], [250, 170, 16], [250, 170, 15], [250, 170, 11], [250, 170, 12], [250, 170, 18], [250, 170, 19], [250, 170, 25], [250, 170, 20], [250, 170, 21], [250, 170, 22], [250, 170, 24]]},
                3: {"name": "Crosswalk", "color": [[140, 140, 200], [200, 128, 128]]},
                4: {"name": "Curb", "color": [[196, 196, 196]]},
                5: {"name": "Sidewalk", "color": [[244, 35, 232]]},

                6: {"name": "Traffic Light", "color": [[250, 170, 30]]},
                7: {"name": "Traffic Sign", "color": [[220, 220, 0]]},

                8: {"name": "Person", "color": [[220, 20, 60]]},
                9: {"name": "Bicyclist", "color": [[255, 0, 0]]},
                10: {"name": "Motorcyclist", "color": [[255, 0, 100]]},

                11: {"name": "Bicycle", "color": [[119, 11, 32], [255, 0, 0]]},
                12: {"name": "Bus", "color": [[0, 60, 100]]},
                13: {"name": "Car", "color": [[0, 0, 142], [0, 0, 90], [0, 0, 110]]},
                14: {"name": "Motorcycle", "color": [[0, 0, 230], [255, 0, 200]]},
                15: {"name": "Truck", "color": [[0, 0, 70]]}
            },
            "A2D2": {
                1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
                2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
                3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
                4: {"name": "Curb", "color": [[128, 128, 0]]},
                5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

                6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
                7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

                8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},
                9: {"name": "Bicyclist", "color": []},
                10: {"name": "Motorcyclist", "color": []},

                11: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
                12: {"name": "Bus", "color": []},
                13: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
                14: {"name": "Motorcycle", "color": [[0, 255, 0], [0, 200, 0], [0, 150, 0]]},
                15: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0], [255, 255, 0], [255, 255, 200]]}
            }
        }

        self.CATEGORIES_COLORS = {
            0: {"name": "Background", "color": [0, 0, 0]},
            1: {"name": "Road", "color": [128, 64, 128]},
            2: {"name": "Lane", "color": [255, 255, 255]},
            3: {"name": "Crosswalk", "color": [200, 128, 128]},
            4: {"name": "Curb", "color": [196, 196, 196]},
            5: {"name": "Sidewalk", "color": [244, 35, 232]},

            6: {"name": "Traffic Light", "color": [250, 170, 30]},
            7: {"name": "Traffic Sign", "color": [220, 220, 0]},

            8: {"name": "Person", "color": [220, 20, 60]},
            9: {"name": "Bicyclist", "color": [255, 0, 0]},
            10: {"name": "Motorcyclist", "color": [255, 0, 100]},

            11: {"name": "Bicycle", "color": [119, 11, 32]},
            12: {"name": "Bus", "color": [0, 60, 100]},
            13: {"name": "Car", "color": [0, 0, 142]},
            14: {"name": "Motorcycle", "color": [0, 0, 230]},
            15: {"name": "Truck", "color": [0, 0, 70]}
        }

    def classes(self):
        return len(self.CATEGORIES[list(self.CATEGORIES.keys())[0]]) + 1

    def labels(self):
        first_cat = list(self.CATEGORIES.keys())[0]
        l = {0: "Background"}
        for i, label in enumerate(map(lambda x: self.CATEGORIES[first_cat][x]["name"], self.CATEGORIES[first_cat]), start=1):
            l[i] = label
        return l

    def colors(self):
        return self.CATEGORIES_COLORS

    def name(self):
        return "MultiDataset"

    def getVistasData(self, dataset_type):
        data_image, data_label = getImagesAndMasksPath("F:\\Mapillary Vistas\\" + dataset_type + "\\images\\", "F:\\Mapillary Vistas\\" + dataset_type + "\\v1.2\labels\\")
        return list(zip(["VISTAS"] * len(data_image), data_image, data_label))

    def getA2D2Data(self, dataset_type):
        dataset_folder = r"F:\\A2D2 Camera Semantic\\" + dataset_type + "\\"

        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(dataset_folder, item) for item in os.listdir(dataset_folder) if os.path.isdir(dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return list(zip(["A2D2"] * len(data_image), data_image, data_label))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.dataset[i: i + self.batch_size]

        # Initialisation des variables de résultat
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size, dtype="uint8")

        # Temp
        ins_255 = np.ones(self.img_size) * 255

        for j, (dataset, image_path, mask_path) in enumerate(batch_input_img_paths):

            ####################
            # CHARGEMENT IMAGE #
            ####################
            x[j] = np.array(load_img(image_path, target_size=self.img_size)) / 255.
            ####################

            ###################
            # CHARGEMENT MASK #
            ###################
            mask = np.array(load_img(mask_path, target_size=self.img_size, color_mode="rgb"))

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for i, id_category in enumerate(self.CATEGORIES[dataset], start=1):
                data_category = self.CATEGORIES[dataset][id_category]

                for color in data_category["color"]:
                    color = tuple(color)

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, color, color)

                    y[j, i] = y[j, i] + (test >= 1)  # Permet d'avoir que des 1 ou 0

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + res

            y[j, 0] = instance == 0

        y = np.moveaxis(y, 1, -1)

        return x, y


class MultiDataset_FOR_DEV_USE_ONLY(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, dataset_type):

        dataset_type = "training" if dataset_type == "train" else ("validation" if dataset_type == "val" else dataset_type)

        self.batch_size = batch_size
        self.img_size = img_size

        # Chargement des données
        self.vistas_dataset = self.getVistasData(dataset_type)[:50]
        self.a2d2_dataset = self.getA2D2Data(dataset_type)[:50]
        self.dataset = self.vistas_dataset + self.a2d2_dataset
        random.shuffle(self.dataset)

        self.CATEGORIES = {
            "VISTAS": {
                1: {"name": "Road", "color": [[128, 64, 128]]},
                2: {"name": "Lane", "color": [[255, 255, 255]]},
                3: {"name": "Crosswalk", "color": [[140, 140, 200], [200, 128, 128]]},
                4: {"name": "Curb", "color": [[196, 196, 196]]},
                5: {"name": "Sidewalk", "color": [[244, 35, 232]]},

                6: {"name": "Traffic Light", "color": [[250, 170, 30]]},
                7: {"name": "Traffic Sign", "color": [[220, 220, 0]]},

                8: {"name": "Person", "color": [[220, 20, 60]]},
                9: {"name": "Bicyclist", "color": [[255, 0, 0]]},
                10: {"name": "Motorcyclist", "color": [[255, 0, 100]]},

                11: {"name": "Bicycle", "color": [[119, 11, 32]]},
                12: {"name": "Bus", "color": [[0, 60, 100]]},
                13: {"name": "Car", "color": [[0, 0, 142]]},
                14: {"name": "Motorcycle", "color": [[0, 0, 230]]},
                15: {"name": "Truck", "color": [[0, 0, 70]]}
            },
            "A2D2": {
                1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
                2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
                3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
                4: {"name": "Curb", "color": [[128, 128, 0]]},
                5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

                6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
                7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

                8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},
                9: {"name": "Bicyclist", "color": []},
                10: {"name": "Motorcyclist", "color": []},

                11: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
                12: {"name": "Bus", "color": []},
                13: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
                14: {"name": "Motorcycle", "color": []},
                15: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0]]}
            }
        }

        self.CATEGORIES_COLORS = {
            0: {"name": "Background", "color": [0, 0, 0]},
            1: {"name": "Road", "color": [128, 64, 128]},
            2: {"name": "Lane", "color": [255, 255, 255]},
            3: {"name": "Crosswalk", "color": [200, 128, 128]},
            4: {"name": "Curb", "color": [196, 196, 196]},
            5: {"name": "Sidewalk", "color": [244, 35, 232]},

            6: {"name": "Traffic Light", "color": [250, 170, 30]},
            7: {"name": "Traffic Sign", "color": [220, 220, 0]},

            8: {"name": "Person", "color": [220, 20, 60]},
            9: {"name": "Bicyclist", "color": [255, 0, 0]},
            10: {"name": "Motorcyclist", "color": [255, 0, 100]},

            11: {"name": "Bicycle", "color": [119, 11, 32]},
            12: {"name": "Bus", "color": [0, 60, 100]},
            13: {"name": "Car", "color": [0, 0, 142]},
            14: {"name": "Motorcycle", "color": [0, 0, 230]},
            15: {"name": "Truck", "color": [0, 0, 70]}
        }

    def classes(self):
        return len(self.CATEGORIES[list(self.CATEGORIES.keys())[0]]) + 1

    def labels(self):
        first_cat = list(self.CATEGORIES.keys())[0]
        l = {0: "Background"}
        for i, label in enumerate(map(lambda x: self.CATEGORIES[first_cat][x]["name"], self.CATEGORIES[first_cat]), start=1):
            l[i] = label
        return l

    def colors(self):
        return self.CATEGORIES_COLORS

    def name(self):
        return "MultiDataset_FOR_DEV_USE_ONLY"

    def getVistasData(self, dataset_type):
        data_image, data_label = getImagesAndMasksPath("F:\\Mapillary Vistas\\" + dataset_type + "\\images\\", "F:\\Mapillary Vistas\\" + dataset_type + "\\v1.2\labels\\")
        return list(zip(["VISTAS"] * len(data_image), data_image, data_label))

    def getA2D2Data(self, dataset_type):
        dataset_folder = r"F:\\A2D2 Camera Semantic\\" + dataset_type + "\\"

        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(dataset_folder, item) for item in os.listdir(dataset_folder) if os.path.isdir(dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return list(zip(["A2D2"] * len(data_image), data_image, data_label))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.dataset[i: i + self.batch_size]

        # Initialisation des variables de résultat
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (self.classes(),) + self.img_size, dtype="uint8")

        # Temp
        ins_255 = np.ones(self.img_size) * 255

        for j, (dataset, image_path, mask_path) in enumerate(batch_input_img_paths):

            ####################
            # CHARGEMENT IMAGE #
            ####################
            x[j] = np.array(load_img(image_path, target_size=self.img_size)) / 255.
            ####################

            ###################
            # CHARGEMENT MASK #
            ###################
            mask = np.array(load_img(mask_path, target_size=self.img_size, color_mode="rgb"))

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for i, id_category in enumerate(self.CATEGORIES[dataset], start=1):
                data_category = self.CATEGORIES[dataset][id_category]

                for color in data_category["color"]:
                    color = tuple(color)

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, color, color)

                    y[j, i] = y[j, i] + (test >= 1)  # Permet d'avoir que des 1 ou 0

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + res

            y[j, 0] = instance == 0

        y = np.moveaxis(y, 1, -1)

        return x, y

if __name__ == "__main__":

    img_size = (480, 720)

    print("\n> Génération du dataset")
    train_gen = MultiDataset_FOR_DEV_USE_ONLY(2, img_size, "train")

    import matplotlib.pyplot as plt
    import tensorflow as tf

    for id in range(len(train_gen)):
        data = train_gen.__getitem__(id)

        for img_i, mask_i in zip(data[0], data[1]):

            nbr_images = 1 if img_i.shape[0] == img_size[0] else img_i.shape[0]

            fig, axs = plt.subplots(nbr_images + 1, 1)

            mask_i = tf.argmax(mask_i, axis=-1)
            mask_i = mask_i[..., tf.newaxis]
            mask_i = tf.keras.preprocessing.image.array_to_img(mask_i)

            if nbr_images == 1:
                axs[0].imshow(img_i)
            else:
                for i in range(nbr_images):
                    axs[i].imshow(img_i[i])

            axs[nbr_images].imshow(mask_i)
            fig.show()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
