import os
import cv2
import time
import numpy as np

from tensorflow import keras, argmax

from utils.argmaxMeanIOU import ArgmaxMeanIOU
from utils.dataset import CATEGORIES_COLORS as OBJECT_SEGMENTATION_COLORS
from utils.dataset_drivable import CATEGORIES_COLORS as LANE_SEGMENTATION_COLORS

from models.all_in_one import Attention_ResUNet_TwoDecoder_TwoOutput

IMG_SIZE = (720, 480)
VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"
MODEL_PATH = r"J:\PROJET\ROAD_SEGMENTATION\trained_models\20220227-180447\AIO-AttentionResUNet-TWO-DECODER-F10_MapillaryVistasDataset_BDD100K-Drivable_384-384_epoch-02_drivable_output_loss-0.00_segmentation_output_loss-327388.81.h5"

OPTIONS = {
    "showRoad": True,
    "showObjects": True,
    "showBackground": True,
}

if __name__ == "__main__":

    # Couleur pour les objets
    objects_values = OBJECT_SEGMENTATION_COLORS.values()
    objects_categories_color = np.zeros((len(objects_values) + 1, 3), dtype=np.uint8)
    for o, data in enumerate(objects_values):
        i = o + 1
        if (i >= 1 and i <= 5 and OPTIONS["showRoad"]) or (i >= 6 and i <= 13 and OPTIONS["showObjects"]) or (i >= 14 and OPTIONS["showBackground"]):
            objects_categories_color[i] = data["color"]

    # Couleur pour les voies de circulation
    lane_values = LANE_SEGMENTATION_COLORS.values()
    lane_categories_color = np.zeros((len(lane_values) + 1, 3), dtype=np.uint8)
    for o, data in enumerate(lane_values):
        i = o + 1
        lane_categories_color[i] = data["color"]

    # Chargement du modèle de la segmentation des objets de la route
    model = keras.models.load_model(MODEL_PATH, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
    model_size = model.get_layer(index=0).input_shape[0][1:-1][::-1]

    # Parcours sur les vidéos
    for video_filename in os.listdir(VIDEO_PATH):

        # Chargement de la vidéo
        filename = os.path.join(VIDEO_PATH, video_filename)
        cap = cv2.VideoCapture(filename)

        new_frame_time = 0
        prev_frame_time = 0

        while(cap.isOpened()):

            ret, frame = cap.read()
            new_frame_time = time.time()

            if not ret:
                break

            img_resized = cv2.resize(frame, model_size, interpolation=cv2.INTER_AREA)

            input_for_model = np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) / 255., axis=0)

            result_segmentation_objects = model.predict(input_for_model)

            result_segmentation_lane = result_segmentation_objects[0][0]
            result_segmentation_objects = result_segmentation_objects[1][0]
            

            # Working on data before argmax
            result_segmentation_lane[result_segmentation_lane < 0.9] = 0
            lane_segmentation = (result_segmentation_lane * 255).astype(np.uint8)
            lane_segmentation[:, :, 0] = 0# 255 - lane_segmentation[:, :, 0]

            lane_segmentation_eroded = lane_segmentation.copy()
            lane_segmentation_eroded[:, :, 0] = 0
            lane_segmentation_eroded = cv2.erode(lane_segmentation_eroded, np.ones((5, 5), np.uint8), iterations=1)

            lane_segmentation = cv2.morphologyEx(lane_segmentation, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))

            # Argmax
            result_segmentation_objects = argmax(result_segmentation_objects, axis=-1)
            # result_segmentation_lane = argmax(result_segmentation_lane, axis=-1)

            # Index --> Couleur

            road_segmentation = np.where(result_segmentation_objects == 1, 255, 0)
            traffic_lane_segmentation = np.where(result_segmentation_objects == 2, 255, 0)
            traffic_crosswalk_segmentation = np.where(result_segmentation_objects == 3, 255, 0)

            road_segmentation = np.array(road_segmentation, dtype=np.uint8)
            traffic_lane_segmentation = np.array(traffic_lane_segmentation, dtype=np.uint8)
            traffic_crosswalk_segmentation = np.array(traffic_crosswalk_segmentation, dtype=np.uint8)


            # Selection des lanes uniquement sur la route du premier model
            lane_segmentation[:,:,0] = np.where(result_segmentation_objects == 1, lane_segmentation[:,:,0], np.zeros_like(lane_segmentation[:,:,0]))
            lane_segmentation[:,:,1] = np.where(result_segmentation_objects == 1, lane_segmentation[:,:,1], np.zeros_like(lane_segmentation[:,:,1]))
            lane_segmentation[:,:,2] = np.where(result_segmentation_objects == 1, lane_segmentation[:,:,2], np.zeros_like(lane_segmentation[:,:,2]))

            result_segmentation_objects = np.expand_dims(result_segmentation_objects, axis=-1)
            object_segmentation = np.squeeze(np.take(objects_categories_color, result_segmentation_objects, axis=0))

            # result_segmentation_lane = np.expand_dims(result_segmentation_lane, axis=-1)
            # lane_segmentation = np.squeeze(np.take(lane_categories_color, result_segmentation_lane, axis=0))

            # IMPROVE: segmentation = cv2.bilateralFilter(segmentation, 10, 75, 75)

            # On calcule le temps nécéssaire
            fps = 1 // (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            # On redimensionne pour la fin
            img_resized = cv2.resize(img_resized, IMG_SIZE, interpolation=cv2.INTER_AREA)
            object_segmentation = cv2.resize(object_segmentation, IMG_SIZE, interpolation=cv2.INTER_AREA)
            lane_segmentation = cv2.resize(lane_segmentation, IMG_SIZE, interpolation=cv2.INTER_AREA)

            lane_segmentation_eroded = cv2.resize(lane_segmentation_eroded, IMG_SIZE, interpolation=cv2.INTER_AREA)

            overlay_segmentation = cv2.addWeighted(object_segmentation, 1, lane_segmentation, 1, 0)

            # On envoie les données
            cv2.imshow("ROAD_IMAGE", img_resized)
            cv2.imshow("SEGMENTATION_IMAGE", cv2.cvtColor(object_segmentation, cv2.COLOR_RGB2BGR))
            cv2.imshow("lane_segmentation_eroded", cv2.cvtColor(lane_segmentation_eroded, cv2.COLOR_RGB2BGR))
            cv2.imshow("OVERLAY", cv2.cvtColor(overlay_segmentation, cv2.COLOR_RGB2BGR))

            cv2.imshow("road_segmentation", cv2.cvtColor(road_segmentation, cv2.COLOR_GRAY2BGR))
            cv2.imshow("traffic_lane_segmentation", cv2.cvtColor(traffic_lane_segmentation, cv2.COLOR_GRAY2BGR))

            # On affiche la vitesse du script
            print(fps)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
