import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from utils.dataset import MultiDataset

from models.bisenetv2 import BiSeNetV2
from models.aunet import Attention_ResUNet

from datetime import datetime
from utils.sanitycheck import SanityCheck
from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

USE_WANDB = True

IMG_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4

if __name__ == '__main__':

    # Generating datasets
    print("\n> Generating datasets")
    train_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "train")
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val")
    test_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "test")

    print("    Training :", len(train_gen), "batchs -", len(train_gen) * BATCH_SIZE, "images")
    print("    Validation :", len(val_gen), "batchs -", len(val_gen) * BATCH_SIZE, "images")



    # Creating model
    print("\n> Creating model")
    model = BiSeNetV2(num_classes=train_gen.classes(), input_shape=IMG_SIZE + (3,))

    optimizer = optimizers.Adam(learning_rate=LR)
    cce = losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer, loss=cce, metrics=['accuracy', ArgmaxMeanIOU(train_gen.classes())])
    # model.summary()



    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        SanityCheck(test_gen, output="trained_models/" + now_str + "/check/", regulator=500, export_files=True, export_wandb=USE_WANDB),
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
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )



    # Weights & Biases - END
    if USE_WANDB:
        run.finish()