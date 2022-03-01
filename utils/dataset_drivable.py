import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

CATEGORIES_COLORS = {
    1: {"name": "Current Lane", "color": [255, 0, 0]},
    2: {"name": "Other Lane", "color": [100, 100, 0]}
}

class DrivableAreaDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def name(self):
        return "BDD100K-Drivable"

    def classes(self):
        return 3

    def colors(self):
        return CATEGORIES_COLORS

    def labels(self):
        first_cat = list(self.colors().keys())[0]
        l = {0: "Background"}
        for i, label in enumerate(map(lambda x: self.colors()[first_cat][x]["name"], self.colors()[first_cat]), start=1):
            l[i] = label
        return l

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = np.array(load_img(path, target_size=self.img_size), dtype=np.uint8) / 255.

        y = np.zeros((self.batch_size,) + (3,) + self.img_size, dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = np.array(load_img(path, target_size=self.img_size, color_mode="grayscale"), dtype=np.uint8)

            # Conversion des couleurs des colormaps (0=0=noir, 130=1=rouge, 174=2=bleu)
            y[j, 0] = (img == 0) * 1.0
            y[j, 1] = (img == 130) * 1.0
            y[j, 2] = (img == 174) * 1.0

        y = np.moveaxis(y, 1, -1)

        z = np.zeros((self.batch_size,) + (17,) + self.img_size)
        z = np.moveaxis(z, 1, -1)

        return x, [y, z] #FIXME: Remove Z and the array


def getImagesAndMasksPath(images_path, masks_path):
    input_train_img_paths = sorted([os.path.join(images_path, fname) for fname in os.listdir(images_path) if fname.endswith(".jpg")])
    target_train_img_paths = sorted([os.path.join(masks_path, fname) for fname in os.listdir(masks_path) if fname.endswith(".png")])
    return input_train_img_paths, target_train_img_paths


if __name__ == '__main__':

    img_size = (384, 384)

    train_input_img_paths, train_target_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/train/", "F:/BDD100K/labels/drivable/colormaps/train/")
    train_gen = DrivableAreaDataset(5, img_size, train_input_img_paths[:10], train_target_img_paths[:10])

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
