import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, argmax
from dataset import MultiDataset_FOR_DEV_USE_ONLY

class SanityCheck(keras.callbacks.Callback):

    id_batch = -1
    epoch = 0

    DEBUG = False

    def __init__(self, dataset, output="./", regulator=200, export_files=True, export_wandb=True):
        super(SanityCheck, self).__init__()
        self.dataset = dataset
        self.data = self.dataset.__getitem__(8)
        self.image_size = [self.data[0][0].shape[0], self.data[0][0].shape[1]]
        self.output = output
        self.regulator = regulator
        self.export_files = export_files
        self.export_wandb = export_wandb

    def on_epoch_end(self, epoch, logs=None):
        # if self.id_batch > 25:
        self.process_test()

        self.id_batch = -1
        self.epoch += 1

    def on_train_batch_end(self, batch, logs=None):
        self.id_batch += 1
        if self.id_batch > 25 and self.id_batch % self.regulator == 0:
            self.process_test()

    def predict_mask(self, img, mask):
        if self.DEBUG == True:
            result = np.repeat(np.expand_dims(np.zeros(self.image_size), axis=2), 2, axis=2)  # Forme du résultat : (X, Y, 2)
        else:
            result = self.model.predict(np.expand_dims(img, axis=0))[0]

        result = np.array(argmax(result, axis=-1), dtype=np.uint8)
        mask = np.array(argmax(mask, axis=-1), dtype=np.uint8)

        return result, mask

    def extract_file(self, result):
        os.makedirs(self.output, exist_ok=True)

        # plt.rcParams["figure.figsize"] = (14, 20)
        fig, axs = plt.subplots(3, len(result))
        fig.suptitle(("MODEL-NAME" if self.DEBUG else self.model.name) + " - I M S")

        colors = self.dataset.colors()
        for i_img, (img_i, mask_i, seg_i) in enumerate(result):

            # Colorisation du masque et du résultat
            mask = np.zeros(img_i.shape, dtype=np.uint8)
            seg = np.zeros(img_i.shape, dtype=np.uint8)
            for categorie in colors.keys():
                mask[mask_i == categorie] = colors[categorie]["color"]
                seg[seg_i == categorie] = colors[categorie]["color"]

            # Affichages des images

            axs[0, i_img].imshow(img_i)
            axs[0, i_img].axis('off')

            axs[1, i_img].imshow(mask)
            axs[1, i_img].axis('off')

            axs[2, i_img].imshow(seg)
            axs[2, i_img].axis('off')

        plt.subplots_adjust(wspace=.05, hspace=.05)
        fig.savefig("%s/%d_%d.png" % (self.output, self.epoch, self.id_batch), dpi=1000, bbox_inches='tight')
        plt.close()

    def extract_wandb(self, result):
        labels = self.dataset.labels()
        wandb_mask_list = list(map(lambda x: wandb.Image(x[0], masks={"prediction": {"mask_data": x[2], "class_labels": labels}, "ground truth": {"mask_data": x[1], "class_labels": labels}}), result))
        wandb.log({"Predictions" : wandb_mask_list})

    def process_test(self):

        result=[]
        imgs, masks=self.data

        for img_i, mask_i in zip(imgs, masks):
            seg, mask_i=self.predict_mask(img_i, mask_i)
            result.append((img_i, mask_i, seg))

        if self.export_files:
            self.extract_file(result)

        if self.export_wandb:
            self.extract_wandb(result)

if __name__ == "__main__":

    # Génération du dataset
    print("\n> Génération du dataset")
    data=MultiDataset_FOR_DEV_USE_ONLY(5, (480, 704), "val")

    sc=SanityCheck(data, output = "TEST-sanity-check/")
    sc.DEBUG=True

    sc.process_test()
