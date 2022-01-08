import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from dataset import MultiDataset

from model import bisenetv2, bisenetv2_DEEPER, bisenetv2_compiled
from aunet import Attention_ResUNet

from datetime import datetime
from sanitycheck import SanityCheck

IMG_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 50

if __name__ == '__main__':

    # Génération des datasets
    print("\n> Génération des datasets")
    train_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "train")
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val")

    print("  len(train_gen) :", len(train_gen), "batchs -", len(train_gen) * BATCH_SIZE, "images")
    print("  len(val_gen) :", len(val_gen), "batchs -", len(val_gen) * BATCH_SIZE, "images")


    # Création du modele
    print("\n> Création du modèle")
    keras.backend.clear_session()

    # Initialisation du modèle
    model = bisenetv2_compiled(Attention_ResUNet, num_classes=train_gen.classes(), input_shape=IMG_SIZE + (3,))

    # from model import ArgmaxMeanIOU
    # model = keras.models.load_model(r"J:\PROJET\IA\BiSeNet-V2\models\20211201-093723\BiSeNet-V2_MultiDataset_480-704_epoch-06_loss-0.13_miou_0.46.h5", custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})

    model.summary()


    # Weights & Biases
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="BiSeNet-V2", entity="nrocher", config={
        "learning_rate": 1e-4,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "dataset": train_gen.name(),
        "model": model.name
    })

    callbacks = [
        SanityCheck(val_gen, output="models/" + now_str + "/check/", regulator=500, export_files=True, export_wandb=True),
        keras.callbacks.ModelCheckpoint("models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}_miou_{val_argmax_mean_iou:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="models/" + now_str + "/logs/", histogram_freq=1),
        WandbCallback()
    ]

    # Entrainement
    print("\n> Entrainement")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )

    # Weights & Biases - END
    run.finish()