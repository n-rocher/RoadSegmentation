import os
import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.argmaxMeanIOU import ArgmaxMeanIOU
from tensorflow import keras, argmax
from utils.dataset import CATEGORIES_COLORS

IMG_SIZE = (720, 480)
OUTPUT_SIZE = (450, 300)
VIDEO_PATH = r"F:/Road Video/25-12-2021/centre ville.mp4"
MODEL_PATH = r"J:/PROJET/IA/RoadSegmentation/models/BiSeNet-V2_MultiDataset_480-704_epoch-12_loss-0.14_miou_0.47.h5"

SHOW_FRAMES = False
EXPORT_GIF = True
MAX_60SEC = True

GIF_DURATION = 40

if __name__ == "__main__":
    segmentation_model = keras.models.load_model(MODEL_PATH, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
    segmentation_model_size = segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]

    video_capture = cv2.VideoCapture(VIDEO_PATH)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    image_lst = []
    image2_lst = []

    i = 0

    with tqdm(total=frame_count, desc="Processing video") as pbar:

        while video_capture.isOpened():
            ret, frame = video_capture.read()

            i += 1

            if not ret or (MAX_60SEC and i > 60 * 10):
                print("Error while reading the video.")
                break

            img_resized = cv2.resize(frame, segmentation_model_size, interpolation=cv2.INTER_AREA)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            result_segmentation = segmentation_model.predict(np.expand_dims(img_resized / 255., axis=0))[0]

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
            image2_lst.append(Image.fromarray(output_image))

            if SHOW_FRAMES:
                cv2.imshow("self.EVT_SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
                cv2.imshow("self.EVT_ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                cv2.imshow("output_image", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) == ord('q'):
                    break

            pbar.update(1)

        pbar.close()

    if EXPORT_GIF:
        print("Printing GIF")
        image2_lst[0].save('./pillow_imagedraw.gif', save_all=True, append_images=image2_lst[1:], optimize=True, loop=0, duration=GIF_DURATION)
        print("Pillow done")
        output_path = os.path.dirname(os.path.abspath(VIDEO_PATH))
        imageio.mimsave('./video.gif', image_lst, fps=40, subrectangles=True)
        print("GIF saved")
