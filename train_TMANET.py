from telnetlib import SE
import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from utils.dataset_sequence import CityscapeSequenceDataset

from models.tma_head import TMAnet

from datetime import datetime
from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

USE_WANDB = False

IMG_SIZE = (384, 384)
BATCH_SIZE = 2
EPOCHS = 500
SEQUENCE_LENGTH = 2
SEQUENCE_DELAY = 2
LR = 0.01
OPTI_MOMENTUM = 0.9

if __name__ == '__main__':

    # Generating datasets
    print("\n> Generating datasets")
    train_gen = CityscapeSequenceDataset(BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_DELAY, IMG_SIZE, "train")
    val_gen = CityscapeSequenceDataset(BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_DELAY, IMG_SIZE, "val")

    print("    Training :", len(train_gen), "batchs")
    print("    Validation :", len(val_gen), "batchs")

    # Creating model
    print("\n> Creating model")
    model = TMAnet(IMG_SIZE, SEQUENCE_LENGTH, train_gen.classes())

    model.load_weights(r"J:\PROJET\ROAD_SEGMENTATION\trained_sequence_models\20220309-125817\TMA-AttentionResUNet-pool_8-F16_CityscapeSequenceDataset-Length-2-Delay-2_384-384_epoch-70_loss-0.61_miou_0.28.h5", by_name=False)
    # model = keras.models.load_model(MODEL_FILE, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})

    # Compiling the model
    learning_rate_fn = optimizers.schedules.PolynomialDecay(LR, len(train_gen), end_learning_rate=0.0001, power=0.9)
    optimizer = optimizers.SGD(learning_rate=learning_rate_fn, momentum=OPTI_MOMENTUM)
    cce = losses.CategoricalCrossentropy(from_logits=True, axis=1)

    model.compile(optimizer, loss=cce, metrics=[ArgmaxMeanIOU(train_gen.classes(), data_format='channels_first')])

    # Model summary
    model.summary(line_length=250)

    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        # SanityCheck(val_gen, output="trained_models/" + now_str + "/check/", regulator=1000, export_files=True, export_wandb=USE_WANDB),
        keras.callbacks.ModelCheckpoint("trained_sequence_models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{loss:.2f}_miou_{argmax_mean_iou:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="trained_sequence_models/" + now_str + "/logs/", histogram_freq=1)
    ]

    # Weights & Biases - Beginning
    if USE_WANDB:
        run = wandb.init(project="Road Segmentation with temporal context", entity="nrocher", config={
            "learning_rate": LR,
            "optizer_momentum": OPTI_MOMENTUM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMG_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "sequence_delay": SEQUENCE_DELAY,
            "dataset": train_gen.name(),
            "model": model.name
        })
        callbacks.append(WandbCallback())

    # Training
    print("\n> Training")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        # validation_data=val_gen,
        use_multiprocessing=True,
        workers=8,
        callbacks=callbacks
    )

    # Weights & Biases - END
    if USE_WANDB:
        run.finish()
