import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from utils.dataset import MultiDataset, MapillaryVistasDataset, NPZDataset

from models.ddrnet_23_slim import DDRNet_23_Slim
from models.bisenetv2 import BiSeNetV2
from models.aunet import Attention_ResUNet, Attention_ResUNet_LIGHTER

from datetime import datetime
from utils.sanitycheck import SanityCheck
from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

USE_WANDB = True

IMG_SIZE = (384, 384)
BATCH_SIZE = 5
EPOCHS = 35
LR = 0.045

MODEL_USED = Attention_ResUNet

A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"

if __name__ == '__main__':

    # Generating datasets
    print("\n> Generating datasets")
    train_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "train", A2D2_FOLDER, VISTAS_FOLDER)
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val", A2D2_FOLDER, VISTAS_FOLDER)

    # train_gen = NPZDataset("DATASET_TRAIN_MapillaryVistasDataset_384-384_CAT-17.npz", BATCH_SIZE)
    # val_gen = NPZDataset("DATASET_VAL_MapillaryVistasDataset_384-384_CAT-17.npz", BATCH_SIZE)

    test_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "test", A2D2_FOLDER, VISTAS_FOLDER)

    print("    Training :", len(train_gen), "batchs -", len(train_gen) * BATCH_SIZE, "images")
    print("    Validation :", len(val_gen), "batchs -", len(val_gen) * BATCH_SIZE, "images")



    # Creating model
    print("\n> Creating model")
    # model = MODEL_USED(num_classes=train_gen.classes(), input_shape=IMG_SIZE + (3,))
 
    MODEL_FILE = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\AttentionResUNet-F16_MultiDataset_384-384_epoch-35_loss-0.28_miou_0.54.h5"
    model = keras.models.load_model(MODEL_FILE, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})


    # optimizer = optimizers.Adam(learning_rate=LR)
    # cce = losses.CategoricalCrossentropy(from_logits=False) # WITH SOFTMAX

    # optimizer = optimizers.SGD(momentum=0.9, lr=LR)
    # cce = losses.CategoricalCrossentropy(from_logits=True) # NO SOFTMAX

    # model.compile(optimizer, loss=cce, metrics=['accuracy', ArgmaxMeanIOU(train_gen.classes())])
    model.summary()


    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        SanityCheck(test_gen, output="trained_models/" + now_str + "/check/", regulator=1000, export_files=True, export_wandb=USE_WANDB),
        keras.callbacks.ModelCheckpoint("trained_models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}_miou_{val_argmax_mean_iou:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="trained_models/" + now_str + "/logs/", histogram_freq=1)
    ]

    if USE_WANDB:
        run = wandb.init(project="Road Segmentation", entity="nrocher", config={
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
            "dataset": train_gen.name(),
            "model": model.name
        })
        callbacks.append(WandbCallback())



    # Training
    print("\n> Training")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        # use_multiprocessing=True,
        # workers=6,
        callbacks=callbacks
    )



    # Weights & Biases - END
    if USE_WANDB:
        run.finish()