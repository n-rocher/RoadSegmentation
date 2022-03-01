from tensorflow import keras

from utils.dataset import MultiDataset, MapillaryVistasDataset, NPZDataset, NPZMultiFileDataset

from models.all_in_one import Attention_ResUNet_TwoDecoder_TwoOutput

from datetime import datetime

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import tensorflow as tf
import tensorflow.keras.metrics as metrics

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 2
LR = 0.8

from utils.dataset_drivable import DrivableAreaDataset, getImagesAndMasksPath

A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"

MODEL_FILE = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\\20220227-170101\AIO-AttentionResUNet-TWO-DECODER-F10_MapillaryVistasDataset_BDD100K-Drivable_384-384_epoch-01_drivable_output_loss-0.58_segmentation_output_loss-0.00.h5"

class ArgmaxMeanIOU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):

        print(y_true.shape, y_pred.shape)

        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)

if __name__ == '__main__':

    # Generating datasets
    print("\n> Generating datasets")
    segmentation_train_gen = MapillaryVistasDataset(BATCH_SIZE, IMG_SIZE, "train", VISTAS_FOLDER)
    segmentation_val_gen = MapillaryVistasDataset(BATCH_SIZE, IMG_SIZE, "val", VISTAS_FOLDER)

    MAX_DATA_TRAINING = 16000
    MAX_DATA_EVALUATION = 500

    # Récupération des paths des fichiers
    print("\n> Récupération des fichiers")
    train_input_img_paths, train_target_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/train/", "F:/BDD100K/labels/drivable/colormaps/train/")
    val_input_img_paths, val_target_img_paths = getImagesAndMasksPath("F:/BDD100K/images/100k/val/", "F:/BDD100K/labels/drivable/colormaps/val/")

    # Génération des datasets
    print("\n> Génération des datasets")
    drivable_train_gen = DrivableAreaDataset(BATCH_SIZE, IMG_SIZE, train_input_img_paths[:MAX_DATA_TRAINING], train_target_img_paths[:MAX_DATA_TRAINING])
    drivable_val_gen = DrivableAreaDataset(BATCH_SIZE, IMG_SIZE, val_input_img_paths[:MAX_DATA_EVALUATION], val_target_img_paths[:MAX_DATA_EVALUATION])


    # Creating model
    print("\n> Creating model")
    model = Attention_ResUNet_TwoDecoder_TwoOutput(drivable_train_gen.classes(), segmentation_train_gen.classes(), input_shape=IMG_SIZE + (3,))
    model.load_weights(MODEL_FILE, by_name=True, skip_mismatch=True)


    optimizer = optimizers.Adam(learning_rate=LR)

    cce_drivable = losses.CategoricalCrossentropy(from_logits=False)
    cce_segmentation = losses.CategoricalCrossentropy(from_logits=False)

    losses = {
        "drivable_output": cce_drivable,
        "segmentation_output": cce_segmentation,
    }


    model.compile(optimizer, loss=losses, metrics=[])
    model.summary()


    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.ModelCheckpoint("trained_models/" + now_str + "/" + model.name + "_" + segmentation_train_gen.name() + "_" + drivable_train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_drivable_output_loss-{drivable_output_loss:.2f}_segmentation_output_loss-{segmentation_output_loss:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="trained_models/" + now_str + "/logs/", histogram_freq=1)
    ]

    relevant_nodes = []
    for v in model._nodes_by_depth.values():
        relevant_nodes += v

    def get_layer_summary_with_connections(layer):
        
        info = {}
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
                connections.append(inbound_layer.name)
                
        return connections


    for o in range(model.layers.index(model.get_layer(name="drivable_start")), 2 + model.layers.index(model.get_layer(name="drivable_output"))):
        print(model.layers[o].name, get_layer_summary_with_connections(model.layers[o]))
    
    exit()

    def changeTrainable(type_s="drivable"):
        for o in range(model.layers.index(model.get_layer(name="drivable_start")), 2 + model.layers.index(model.get_layer(name="drivable_output")), 2):
            i = o + 0
            model.layers[i].trainable = True if type_s == "drivable" else not False

        for o in range(model.layers.index(model.get_layer(name="drivable_start")), 2 + model.layers.index(model.get_layer(name="drivable_output")), 2):
            i = o + 1
            model.layers[i].trainable = True if type_s == "segmentation" else not False


    for i in range(10):

        # Training drivable
        print("\n> Training drivable")
        changeTrainable("drivable")
        model.fit(
            drivable_train_gen,
            epochs=EPOCHS,
            use_multiprocessing=True,
            workers=6,
            callbacks=callbacks
        )

        # Training segmentation
        print("\n> Training segmentation")
        changeTrainable("segmentation")
        model.fit(
            segmentation_train_gen,
            epochs=EPOCHS,
            use_multiprocessing=True,
            workers=6,
            callbacks=callbacks
        )

        LR = LR / 2

        if LR < 0.0001:
            LR = 0.0001

        