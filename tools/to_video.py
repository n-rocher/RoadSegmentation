import os
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob
from utils.argmaxMeanIOU import ArgmaxMeanIOU
from tensorflow import keras, argmax
from utils.dataset import CATEGORIES_COLORS

IMG_SIZE = (720, 480)
OUTPUT_SIZE = (1280, 720)

VIDEO_PATH = r"F:\ROAD_VIDEO\CLIP\*"
MODEL_PATH = r"./trained_models/AttentionResUNet-F16_MultiDataset_384-384_epoch-60_loss-0.31_miou_0.54.h5"

SHOW_FRAMES = True
EXPORT_VIDEO = True
MAX_BATCH_SIZE = 10
MAX_60SEC = True

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

    for model_path in glob(MODEL_PATH):

        segmentation_model = keras.models.load_model(model_path, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
        segmentation_model_size = segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]

        for video_path in glob(VIDEO_PATH):

            video_name = os.path.basename(video_path)
            model_name = os.path.basename(model_path)

            video_capture = cv2.VideoCapture(video_path)
            video_result = cv2.VideoWriter('./video-output/' + video_name + "---" + model_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, OUTPUT_SIZE)

            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            i = 0

            img_for_process = []

            with tqdm(total=frame_count, desc="Video : " + video_name) as pbar:

                while video_capture.isOpened():
                    ret, frame = video_capture.read()

                    i += 1

                    if not ret or (MAX_60SEC and i > (120 * 10)):
                        print("Error while reading the video.")
                        pbar.close()
                        break

                    img_resized = cv2.resize(frame, segmentation_model_size, interpolation=cv2.INTER_AREA)
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

                    img_for_process.append(img_resized)

                    if len(img_for_process) == MAX_BATCH_SIZE:

                        img_for_process = np.array(img_for_process)

                        result_batch = segmentation_model.predict(img_for_process / 255.)

                        for j in range(MAX_BATCH_SIZE):

                            img_resized = img_for_process[j]
                            result_segmentation = result_batch[j]

                            # Argmax
                            result_segmentation = argmax(result_segmentation, axis=-1)
                            argmax_result_segmentation = np.expand_dims(result_segmentation, axis=-1)
                            segmentation = np.squeeze(np.take(categories_color, argmax_result_segmentation, axis=0))

                            segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, (10, 10))

                            if segmentation_model_size != OUTPUT_SIZE:
                                img_resized = cv2.resize(img_resized, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
                                segmentation = cv2.resize(segmentation, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

                            overlay_segmentation = cv2.addWeighted(img_resized, 0.3, segmentation, 0.9, 0)
                            output_image = cv2.hconcat([img_resized, overlay_segmentation])

                            video_result.write(cv2.cvtColor(overlay_segmentation, cv2.COLOR_RGB2BGR))

                            if SHOW_FRAMES:
                                cv2.imshow("self.EVT_SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
                                cv2.imshow("self.EVT_ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                                cv2.imshow("output_image", cv2.cvtColor(overlay_segmentation, cv2.COLOR_RGB2BGR))

                                if cv2.waitKey(1) == ord('q'):
                                    pbar.close()
                                    break

                            pbar.update(1)

                        img_for_process = []

                pbar.close()
                video_result.release()
