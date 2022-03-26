import os
import cv2
import sys
import time
import numpy as np

from tensorflow import keras, argmax
import torch.nn.functional as F
import torch
from utils.dataset_sequence import CITYSCAPE_CATEGORIES
from utils.argmaxMeanIOU import ArgmaxMeanIOU

from models.tmanet import TMA_ResUnet

VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"
MODEL_FILE = r"J:\PROJET\ROAD_SEGMENTATION\trained_sequence_models\20220309-125817\TMA-AttentionResUNet-pool_8-F16_CityscapeSequenceDataset-Length-2-Delay-2_384-384_epoch-26_loss-0.74_miou_0.22.h5"


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    return F.interpolate(input, size, scale_factor, mode, align_corners)


if __name__ == "__main__":

    values = CITYSCAPE_CATEGORIES.values()
    categories_color = np.zeros((len(values), 3), dtype=np.uint8)
    for i, data in enumerate(values):
        categories_color[i] = data["color"]

    SEQUENCE_LENGTH = 2
    SEQUENCE_DELAY = 2
    OUTPUT_SIZE = (640, 480)

    model = TMA_ResUnet((384, 384), SEQUENCE_LENGTH, len(values)) # keras.models.load_model(MODEL_FILE, compile=False)

    INPUT_SIZE = model.input[0].shape[-2:]

    # Boucle des vidéos
    for video_filename in os.listdir(VIDEO_PATH):

        filename = os.path.join(VIDEO_PATH, video_filename)

        cap = cv2.VideoCapture(filename)

        new_frame_time = 0
        prev_frame_time = 0

        previous_frames = np.zeros((SEQUENCE_LENGTH, ) + (3,) + INPUT_SIZE[::-1], dtype=np.float64)

        while(cap.isOpened()):

            ret, frame = cap.read()
            new_frame_time = time.time()

            if not ret:
                break


            NORMALIZATION_MEAN = [123.675, 116.28, 103.53]
            NORMALIZATION_STD = [58.395, 57.12, 57.375]

            img_for_model = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_AREA)
            img_for_model = (img_for_model - NORMALIZATION_MEAN)  / NORMALIZATION_STD

            # channel_last -> channel_first 
            img_for_model = np.rollaxis(img_for_model, 2, 0)

            # Conversion en batch de 1
            img_for_model = np.expand_dims(img_for_model, axis=0)
            sequence_imgs = np.expand_dims(previous_frames, axis=0)

            # Prediction
            result_segmentation = model.predict([img_for_model, sequence_imgs])

            # Argmax
            argmax_result_segmentation = argmax(np.squeeze(result_segmentation), axis=0)

            # Index --> Couleur
            argmax_result_segmentation = np.expand_dims(argmax_result_segmentation, axis=-1)
            segmentation = np.squeeze(np.take(categories_color, argmax_result_segmentation, axis=0))

            # Saving last item
            previous_frames = np.roll(previous_frames, 1, axis=0)
            previous_frames[0] = img_for_model[0]

            # On calcule le temps nécéssaire
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            print(fps)

            # Affichage des résultats

            # On redimenssione les résultats pour les afficher correctement
            cv2.imshow("Frame", cv2.resize(frame, OUTPUT_SIZE))

            for i in range(SEQUENCE_LENGTH):
                cv2.imshow('Frame t-' + str(i + 1), cv2.resize(np.rollaxis(previous_frames[i], 0, 3), OUTPUT_SIZE))

            cv2.imshow("Segmentation", cv2.cvtColor(cv2.resize(segmentation, OUTPUT_SIZE), cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
