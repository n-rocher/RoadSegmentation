from utils.dataset import MultiDataset

from utils.argmaxMeanIOU import ArgmaxMeanIOU

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity


MODEL_FILE = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\20220109-234648\AttentionResUNet-F16_MultiDataset_512-512_epoch-26_loss-0.21_miou_0.55.h5"

IMG_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 4
LR = 1e-4

if __name__ == '__main__':

    # Generating datasets
    print("\n> Generating datasets")
    val_gen = MultiDataset(BATCH_SIZE, IMG_SIZE, "val")

    print("    Validation USED FOR TRAINING THE OPTIMISATION :", len(val_gen), "batchs -", len(val_gen) * BATCH_SIZE, "images")



    # Loading model
    print("\n> Loading model")
    loaded_model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})


    end_step = np.ceil(1.0 * len(val_gen) / BATCH_SIZE).astype(np.int32) * EPOCHS
    print("end_step", end_step)


    new_pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.90, begin_step=0, end_step=end_step, frequency=100)
    }

    new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
    new_pruned_model.summary()

    optimizer = optimizers.Adam(learning_rate=LR)
    cce = losses.CategoricalCrossentropy(from_logits=True)


    new_pruned_model.compile(optimizer, loss=cce, metrics=['accuracy', ArgmaxMeanIOU(val_gen.classes())])

    callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir="./logdir", profile_batch=0)
    ]

    new_pruned_model.fit(
        val_gen,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks)

    new_pruned_model.save("new_pruned_model.h5")
    
    final_model = sparsity.strip_pruning(new_pruned_model)
    final_model.summary()

    final_model.save("final_model_optimized.h5")
