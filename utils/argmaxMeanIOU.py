import tensorflow as tf
import tensorflow.keras.metrics as metrics

class ArgmaxMeanIOU(metrics.MeanIoU):

    def __init__(self, num_classes, data_format='channels_last'):
        super().__init__(num_classes)
        
        self.axis = -1

        if data_format == "channels_first":
            self.axis = 1

    def update_state(self, y_true, y_pred, sample_weight=None):

        return super().update_state(tf.argmax(y_true, axis=self.axis), tf.argmax(y_pred, axis=self.axis), sample_weight)