import os
import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
from utils.argmaxMeanIOU import ArgmaxMeanIOU
from tensorflow import keras, argmax
from utils.dataset import CATEGORIES_COLORS
import tensorflow as tf


IMG_SIZE = (720, 480)
OUTPUT_SIZE = (450, 300)

VIDEO_PATH = r"F:\Road Video\Clip\*"
MODEL_PATH = r"model.tflite"

SHOW_FRAMES = False
EXPORT_GIF = True
MAX_60SEC = True

GIF_DURATION = 40

if __name__ == "__main__":

    video_path = glob(VIDEO_PATH)[0]

    interpreter = tf.lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("input_details", input_details)
    print("output_details", output_details)

    video_name = os.path.basename(video_path)
    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("Error while reading the video.")
            break

        print(input_details[0]['shape'])
        print(input_details[0]['shape'][1:-1])

        img_resized = cv2.resize(frame, input_details[0]['shape'][1:-1], interpolation=cv2.INTER_AREA)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        image = np.array([img_resized/255.0], dtype=np.float32)

        print("set_tensor")
        interpreter.set_tensor(input_details[0]['index'], image)
        print("invoke")
        interpreter.invoke()

        print("get_tensor")
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        print(output_data.shape)

        # Argmax
        result_segmentation = argmax(output_data, axis=-1)
        segmentation = np.zeros(result_segmentation.shape + (3,), dtype=np.uint8)
        for categorie in CATEGORIES_COLORS.keys():
            segmentation[result_segmentation == categorie] = CATEGORIES_COLORS[categorie]["color"]

        overlay_segmentation = cv2.addWeighted(img_resized, 0.7, segmentation, 0.7, 0.52)
        output_image = cv2.hconcat([img_resized, segmentation, overlay_segmentation])

        cv2.imshow("self.EVT_SEGMENTATION_IMAGE", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR))
        cv2.imshow("self.EVT_ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        cv2.imshow("output_image", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) == ord('q'):
            break