from tensorflow import keras

from models.aunet import Attention_ResUNet

from datetime import datetime
from utils.sanitycheck import SanityCheck
from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

from tensorflow.keras import models, layers

from utils.dataset import MultiDataset

IMG_SIZE = (384, 384)
BATCH_SIZE = 5
EPOCHS = 35
LR = 0.045


MODEL_PATH = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\AttentionResUNet-F16_MultiDataset_384-384_epoch-60_loss-0.31_miou_0.54.h5"


A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"
    

if __name__ == '__main__':

    # Génération des datasets
    print("\n> Génération des datasets")
    train_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "train", A2D2_FOLDER, VISTAS_FOLDER)
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val", A2D2_FOLDER, VISTAS_FOLDER)

    print("\n> Loading model")

    model = Attention_ResUNet(num_classes=17, input_shape=IMG_SIZE + (3,))
    model.load_weights(MODEL_PATH)

    conv_final = layers.Activation('softmax', name="softmax")(model.layers[-1].output)

    model = models.Model(model.input, conv_final, name="AttentionResUNet-WITH-SOFTMAX")
   
    optimizer = optimizers.Adam(learning_rate=LR)
    cce = losses.CategoricalCrossentropy(from_logits=False)

    model.compile(optimizer, loss=cce, metrics=[ArgmaxMeanIOU(train_gen.classes())])
    model.summary()


    # Callbacks
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        keras.callbacks.ModelCheckpoint("trained_models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{loss:.2f}_miou_{argmax_mean_iou:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="trained_models/" + now_str + "/logs/", histogram_freq=1)
    ]

    # Training
    print("\n> Training")
    model.fit(
        val_gen,
        epochs=EPOCHS,
        # validation_data=val_gen,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )

