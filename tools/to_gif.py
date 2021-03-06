import os
from tkinter.tix import MAX
import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from utils.argmaxMeanIOU import ArgmaxMeanIOU
from tensorflow import keras, argmax
from utils.dataset import CATEGORIES_COLORS

IMG_SIZE = (720, 480)
OUTPUT_SIZE = (450, 300)

VIDEO_PATH = r"F:\ROAD_VIDEO\CLIP\*"
MODEL_PATH = r"./trained_models/AttentionResUNet-F16_MultiDataset_384-384_epoch-60_loss-0.31_miou_0.54.h5"

SHOW_FRAMES = False
EXPORT_GIF = True
MAX_60SEC = True
MAX_BATCH_SIZE = 25

GIF_DURATION = 40

if __name__ == "__main__":

    for model_path in glob(MODEL_PATH):

        segmentation_model = keras.models.load_model(model_path, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
        segmentation_model_size = segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]

        # video_path = r"F:\ROAD_VIDEO\CLIP\ombre complexe + croisement de route.mp4"
        for video_path in glob(VIDEO_PATH):

            video_name = os.path.basename(video_path)
            model_name = os.path.basename(model_path)

            video_capture = cv2.VideoCapture(video_path)

            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            image_lst = []

            i = 0

            img_for_process = []

            with tqdm(total=frame_count, desc="Video : " + video_name) as pbar:

                while video_capture.isOpened():
                    ret, frame = video_capture.read()

                    i += 1

                    if not ret or (MAX_60SEC and i > 120 * 10):
                        print("Error while reading the video.")
                        break

                    img_resized = cv2.resize(frame, segmentation_model_size, interpolation=cv2.INTER_AREA)
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

                    img_for_process.append(img_resized)

                    if len(img_for_process) == MAX_BATCH_SIZE:

                        img_for_process = np.array(img_for_process)

                        result_batch = segmentation_model.predict(img_for_process  / 255. )

                        for i in range(MAX_BATCH_SIZE):

                            img_resized = img_for_process[i]
                            result_segmentation = result_batch[i]

                            # Argmax
                            result_segmentation = argmax(result_segmentation, axis=-1)
                            segmentation = np.zeros(result_segmentation.shape + (3,), dtype=np.uint8)
                            for categorie in CATEGORIES_COLORS.keys():
                                segmentation[result_segmentation == categorie] = CATEGORIES_COLORS[categorie]["color"]

                            if segmentation_model_size != OUTPUT_SIZE:
                                img_resized = cv2.resize(img_resized, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
                                segmentation = cv2.resize(segmentation, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)

                            overlay_segmentation = cv2.addWeighted(img_resized, 0.7, segmentation, 0.7, 0.52)
                            output_image = cv2.hconcat([img_resized, segmentation, overlay_segmentation])

                            image_lst.append(output_image)

                            if SHOW_FRAMES:
                                cv2.imshow("self.EVT_SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
                                cv2.imshow("self.EVT_ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                                cv2.imshow("output_image", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

                                if cv2.waitKey(1) == ord('q'):
                                    break

                            pbar.update(1)

                        img_for_process = []

                pbar.close()

            if EXPORT_GIF:
                print("Printing GIF")
                imageio.mimsave('./image/' + video_name + "---" + model_name + '.gif', image_lst, fps=40, subrectangles=True)
                print("GIF saved")