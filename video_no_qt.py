import os
import cv2
import sys
import time
import numpy as np

from tensorflow import keras, argmax

from utils.argmaxMeanIOU import ArgmaxMeanIOU
from utils.dataset import CATEGORIES_COLORS

IMG_SIZE = (720, 480)
VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"
MODEL_PATH = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\AttentionResUNet-F16_MultiDataset_512-512_epoch-26_loss-0.21_miou_0.55.h5"

OPTIONS = {
    "showRoad": True,
    "showObjects": True,
    "showBackground": True,
}

if __name__ == "__main__":

    values = CATEGORIES_COLORS.values()
    categories_color = np.zeros((len(values) + 1, 3), dtype=np.uint8)
    for o, data in enumerate(values):
        i = o + 1
        if (i >= 1 and i <= 5 and OPTIONS["showRoad"]) or (i >= 6 and i <= 13 and OPTIONS["showObjects"]) or (i >= 14 and OPTIONS["showBackground"]):
            categories_color[i] = data["color"]

    for video_filename in os.listdir(VIDEO_PATH):

        filename = os.path.join(VIDEO_PATH, video_filename)

        segmentation_model = keras.models.load_model(MODEL_PATH, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
        segmentation_model_size = segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]

        cap = cv2.VideoCapture(filename)

        new_frame_time = 0
        prev_frame_time = 0

        while(cap.isOpened()):

            ret, frame = cap.read()
            new_frame_time = time.time()

            if not ret:
                continue

            img_resized = cv2.resize(frame, segmentation_model_size, interpolation=cv2.INTER_AREA)

            result_segmentation = segmentation_model.predict(np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) / 255., axis=0))[0]
            result_segmentation_with_temp = result_segmentation

            # Argmax
            argmax_result_segmentation = argmax(result_segmentation_with_temp, axis=-1)

            # Index --> Couleur
            argmax_result_segmentation = np.expand_dims(argmax_result_segmentation, axis=-1)
            segmentation = np.squeeze(np.take(categories_color, argmax_result_segmentation, axis=0))

            # On redimenssione les résultats pour les afficher correctement
            if segmentation_model_size != (640, 480):
                img_resized = cv2.resize(img_resized, (640, 480), interpolation=cv2.INTER_AREA)
                segmentation = cv2.resize(segmentation, (640, 480), interpolation=cv2.INTER_AREA)

            # On calcule le temps nécéssaire
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # On envoie les données
            cv2.imshow("ROAD_IMAGE", img_resized)
            cv2.imshow("SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))

            print(fps)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
