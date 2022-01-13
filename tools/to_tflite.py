from utils.argmaxMeanIOU import ArgmaxMeanIOU
import tensorflow as tf


MODEL_FILE = r"./trained_models/AttentionResUNet-F16_MultiDataset_512-512_epoch-26_loss-0.21_miou_0.55.h5"

if __name__ == '__main__':

    # Loading model
    print("\n> Loading model")
    loaded_model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})


    # Post training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)