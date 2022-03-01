import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

DRIVABLE_CATEGORIES_COLORS = {
    1: {"name": "Direct", "color": [255, 0, 0]},
    2: {"name": "Alternative", "color": [100, 100, 0]}
}

SEGMENTATION_CATEGORIES_COLORS = {
    1: {"name": "road", "color": [75, 75, 75]},
    2: {"name": "sidewalk", "color": [244, 35, 232]},
    3: {"name": "building", "color": [241, 230, 255]},
    4: {"name": "wall", "color": [200, 200, 200]},
    5: {"name": "fence", "color": [100, 100, 100]},
    6: {"name": "pole", "color": [255, 192, 203]},
    7: {"name": "traffic_light", "color": [250, 170, 30]},
    8: {"name": "traffic_sign", "color": [255, 255, 0]},
    9: {"name": "vegetation", "color": [107, 142, 35]},
    10: {"name": "terrain", "color": [184, 134, 11]},
    11: {"name": "sky", "color": [135, 206, 255]},
    12: {"name": "person", "color": [255, 0, 0]},
    13: {"name": "rider", "color": [150, 0, 0]},
    14: {"name": "car", "color": [0, 255, 142]},
    15: {"name": "truck", "color": [75, 10, 170]},
    16: {"name": "bus", "color": [255, 15, 147]},
    17: {"name": "train", "color": [0, 150, 150]},
    18: {"name": "motorcycle", "color": [0, 0, 230]},
    19: {"name": "bicycle", "color": [88, 41, 0]}
}

class BDD100KDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, drivable_target_img_paths, segmentation_target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.drivable_target_img_paths = drivable_target_img_paths
        self.segmentation_target_img_paths = segmentation_target_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def name(self):
        return "BDD100K-Segmentation-Drivable"

    def colors(self):
        return [DRIVABLE_CATEGORIES_COLORS, SEGMENTATION_CATEGORIES_COLORS]

    def classes(self):
        return [len(obj.keys()) + 1 for obj in self.colors()]

    def labels(self):
        labels = []
        for data in self.colors():
            first_cat = list(data.keys())[0]
            l = {0: "Background"}
            for i, label in enumerate(map(lambda x: data[first_cat][x]["name"], data[first_cat]), start=1):
                l[i] = label
            labels.append(l)
        return labels

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_drivable_target_img_paths = self.drivable_target_img_paths[i: i + self.batch_size]
        batch_segmentation_target_img_paths = self.segmentation_target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + (3,) + self.img_size, dtype="uint8")
        z = np.zeros((self.batch_size,) + (3,) + self.img_size, dtype="uint8")

        for j, (image, drivable, segmentation) in enumerate(zip(batch_input_img_paths, batch_drivable_target_img_paths, batch_segmentation_target_img_paths)):
            x[j] = np.array(load_img(image, target_size=self.img_size), dtype=np.uint8) / 255.

            drivable = np.array(load_img(drivable, target_size=self.img_size, color_mode="grayscale"), dtype=np.uint8)
            segmentation = np.array(load_img(segmentation, target_size=self.img_size, color_mode="grayscale"), dtype=np.uint8)

            print(np.unique(drivable))
            print(np.unique(segmentation))

            # Conversion des couleurs des colormaps (0=0=noir, 130=1=rouge, 174=2=bleu)
            y[j, 0] = (drivable == 0) * 1.0
            y[j, 1] = (drivable == 130) * 1.0
            y[j, 2] = (drivable == 174) * 1.0

        y = np.moveaxis(y, 1, -1)
        z = np.moveaxis(z, 1, -1)

        return x, y, z


def getImagesAndMasksPath(images_path, drivable_masks_path, segmentation_masks_path):
    input_train_img_paths = sorted([os.path.join(images_path, fname) for fname in os.listdir(images_path) if fname.endswith(".jpg")])
    target_drivable_img_paths = sorted([os.path.join(drivable_masks_path, fname) for fname in os.listdir(drivable_masks_path) if fname.endswith(".png")])
    target_segmentation_img_paths = sorted([os.path.join(segmentation_masks_path, fname) for fname in os.listdir(segmentation_masks_path) if fname.endswith(".png")])
    
    print(len(input_train_img_paths), len(target_drivable_img_paths), len(target_segmentation_img_paths))

    assert len(input_train_img_paths) == len(target_drivable_img_paths) and len(target_drivable_img_paths) == len(target_segmentation_img_paths), "Il n'y a pas le meme nombre de masque"

    return input_train_img_paths, target_drivable_img_paths, target_segmentation_img_paths


if __name__ == '__main__':

    img_size = (384, 384)

    train_input_img_paths, target_drivable_img_paths, target_segmentation_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/train/", "F:/BDD100K/labels/drivable/masks/train/", "F:/BDD100K/labels/sem_seg/masks/train/")
    train_gen = BDD100KDataset(5, img_size, train_input_img_paths[:10], target_drivable_img_paths[:10], target_segmentation_img_paths[:10])

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
