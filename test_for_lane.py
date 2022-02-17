from tensorflow import keras

from models.aunet import Attention_ResUNet

from datetime import datetime
from utils.sanitycheck import SanityCheck
from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import backend as K

from utils.dataset_drivable import DrivableAreaDataset, getImagesAndMasksPath

IMG_SIZE = (384, 384)
BATCH_SIZE = 5
EPOCHS = 35
LR = 0.045


A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"
MODEL_PATH = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\AttentionResUNet-F16_MultiDataset_384-384_epoch-35_loss-0.28_miou_0.54.h5"

if __name__ == '__main__':

    MAX_DATA_TRAINING = 200000000
    MAX_DATA_EVALUATION = 500

    # Récupération des paths des fichiers
    print("\n> Récupération des fichiers")
    train_input_img_paths, train_target_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/train/", "F:/BDD100K/labels/drivable/colormaps/train/")
    val_input_img_paths, val_target_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/val/", "F:/BDD100K/labels/drivable/colormaps/val/")

    # Génération des datasets
    print("\n> Génération des datasets")
    train_gen = DrivableAreaDataset(BATCH_SIZE, IMG_SIZE, train_input_img_paths[:MAX_DATA_TRAINING], train_target_img_paths[:MAX_DATA_TRAINING])
    val_gen = DrivableAreaDataset(BATCH_SIZE, IMG_SIZE, val_input_img_paths[:MAX_DATA_EVALUATION], val_target_img_paths[:MAX_DATA_EVALUATION])


    print("\n> Loading model")

    model = Attention_ResUNet(num_classes=17, input_shape=IMG_SIZE + (3,))
    model.load_weights(MODEL_PATH)

    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    conv_final = layers.Conv2D(filters=train_gen.classes() * 2, kernel_size=(3, 3), padding='same', name="aa")(model.layers[-4].output)
    conv_final = layers.BatchNormalization(name="bb")(conv_final)
    conv_final = layers.Activation('relu', name="cc")(conv_final)

    conv_final = layers.Conv2D(filters=train_gen.classes(), kernel_size=(3, 3), padding='same', name="dd")(conv_final)

    model = models.Model(model.input, conv_final, name="AttentionResUNet-FOR-LANE")
   
    optimizer = optimizers.Adam(learning_rate=LR)
    cce = losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss=cce, metrics=['accuracy', ArgmaxMeanIOU(train_gen.classes())])
    model.summary()


    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        SanityCheck(val_gen, output="trained_models/" + now_str + "/check/", regulator=1000, export_files=True, export_wandb=False),
        keras.callbacks.ModelCheckpoint("trained_models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}_miou_{val_argmax_mean_iou:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="trained_models/" + now_str + "/logs/", histogram_freq=1)
    ]

    # Training
    print("\n> Training")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )

