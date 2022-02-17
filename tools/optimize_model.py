from utils.dataset import MultiDataset

from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

from datetime import datetime

MODEL_FILE = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\AttentionResUNet-F16_MultiDataset_384-384_epoch-35_loss-0.28_miou_0.54.h5"

IMG_SIZE = (384, 384)
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3

if __name__ == '__main__':

    A2D2_FOLDER = r"F:\\A2D2 Camera Semantic\\"
    VISTAS_FOLDER = r"F:\\Mapillary Vistas\\"

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")


    # Generating datasets
    print("\n> Generating datasets")
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val", A2D2_FOLDER, VISTAS_FOLDER)

    print("    Validation USED FOR TRAINING THE OPTIMISATION :", len(val_gen), "batchs -", len(val_gen) * BATCH_SIZE, "images")

    # Loading model
    print("\n> Loading model")
    loaded_model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})

    end_step = np.ceil(1.0 * len(val_gen) / BATCH_SIZE).astype(np.int32) * EPOCHS
    print("end_step", end_step)

    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=0, end_step=end_step)
    }

    new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
    new_pruned_model.summary()

    optimizer = optimizers.Adam(learning_rate=LR)
    cce = losses.CategoricalCrossentropy(from_logits=True)

    new_pruned_model.compile(optimizer, loss=cce, metrics=['accuracy', ArgmaxMeanIOU(val_gen.classes())])

    final_model = None

    def saveModel(epoch, _):
        global final_model
        final_model = sparsity.strip_pruning(new_pruned_model)
        file_name = "optimized_models/" + now_str + "/" + new_pruned_model.name + "-OPTIMIZED_" + val_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-" + str(epoch) + ".h5"
        final_model.save(file_name, include_optimizer=False)

    callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir="./optimized_logdir", profile_batch=0),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=saveModel)
    ]

    new_pruned_model.fit(
        val_gen,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks)
    
    final_model.summary()
