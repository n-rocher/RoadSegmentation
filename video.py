import os
import cv2
import sys
import time
import numpy as np

from tensorflow import keras, argmax

from utils.argmaxMeanIOU import ArgmaxMeanIOU
from utils.dataset import CATEGORIES_COLORS

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QCheckBox, QSizePolicy, QVBoxLayout, QWidget)

TRAFFIC_SIGN_DATASET = {
    1: "Virage à droite",
    100: "Sens unique (droit)",
    107: "Zone 30",
    108: "Fin zone 30",
    109: "Passage pour piétons",
    11: "Ralentisseur simple",
    12: "Ralentisseur double",
    125: "Ralentisseur",
    13: "Route glissante",
    140: "Direction",
    15: "Chute de pierres",
    16: "Passage pour piétons",
    17: "Enfants (école)",
    2: "Virage à gauche",
    23: "Intersection",
    24: "Intersection avec une route",
    25: "Rond-point",
    3: "Double virage (gauche)",
    32: "Autres dangers",
    35: "Céder le passage",
    36: "Stop",
    37: "Route prioritaire",
    38: "Fin route prioritaire",
    39: "Priorité au trafic en sens inverse",
    4: "Double virage (droite)",
    40: "Priorité au trafic en sens inverse",
    41: "Sens interdit",
    51: "Virage à gauche interdit",
    52: "Virage à droite interdit",
    53: "Demi-tour interdit",
    54: "Dépassement interdit",
    55: "Dépassement interdit aux véhicules de transport de marchandises",
    57: "Vitesse maximale 20",
    59: "Vitesse maximale 30",
    60: "Vitesse maximale 40",
    61: "Vitesse maximale 50",
    62: "Vitesse maximale 60",
    63: "Vitesse maximale 70",
    64: "Vitesse maximale 80",
    65: "Vitesse maximale 90",
    66: "Vitesse maximale 100",
    67: "Vitesse maximale 110",
    68: "Vitesse maximale 120",
    7: "Rétrécissement de la chaussée",
    80: "Direction - Tout droit",
    81: "Direction - Droite",
    82: "Direction - Gauche",
    83: "Direction - Tout droit ou à droite",
    84: "Direction - Tout droit ou à gauche",
    85: "Direction - Tourner à droite",
    86: "Direction - Tourner à gauche",
    87: "Passer à droite",
    88: "Passer à gauche"
}

TRAFFIC_SIGN_DATASET_VALUES = list(TRAFFIC_SIGN_DATASET.values())
TRAFFIC_SIGN_DATASET_KEYS = list(TRAFFIC_SIGN_DATASET.keys())
TRAFFIC_SIGN_DATASET_IMAGE_FOLDER = "J:/PROJET/TRAFFIC_SIGN_RECOGNITION/data/logo/"
TRAFFIC_SIGN_DATASET_IMAGE = list(map(lambda x: cv2.resize(cv2.imread(TRAFFIC_SIGN_DATASET_IMAGE_FOLDER + str(x) + ".png", cv2.IMREAD_UNCHANGED), (50, 50)), TRAFFIC_SIGN_DATASET_KEYS))

IMG_SIZE = (720, 480)
VIDEO_PATH = r"F:\ROAD_VIDEO\Clip"

BOUNDING_BOX_PADDING = 5
TIME_PERSISTANT = 5

def copyTrafficSign(image, trafficSign, x, y):

    assert x >= 0 and y >= 0, "x et y doivent etre supérieur à 0"

    i_h, i_w, _ = image.shape
    t_h, t_w, _ = trafficSign.shape

    # Récupération logo
    temp_w = i_w - x
    t_w = t_w if temp_w >= t_w else t_w - temp_w

    temp_h = i_h - y
    t_h = t_h if temp_h >= t_h else t_h - temp_h

    logo = trafficSign[0:t_h, 0:t_w, :3]
    mask = trafficSign[0:t_h, 0:t_w, 3]

    # Récupération image
    sous_parties_image = image[y:y + t_h, x:x + t_w, :]

    # Ajout des masques

    fg = cv2.bitwise_or(logo, logo, mask=mask)
    bg = cv2.bitwise_or(sous_parties_image, sous_parties_image, mask=cv2.bitwise_not(mask))

    # Assemblage final
    final = cv2.bitwise_or(fg, bg)
    image[y:y + t_h, x:x + t_w, :] = final

    return image


class Thread(QThread):
    EVT_ROAD_IMAGE = Signal(QImage)
    EVT_SEGMENTATION_IMAGE = Signal(QImage)
    EVT_FPS = Signal(int)

    segmentation_model = None
    segmentation_model_size = None

    traffic_sign_recognition_model = None
    traffic_sign_recognition_model_size = None

    options = {
        "showRoad": True,
        "showObjects": True,
        "showBackground": True,
        "useTimeConsistency": True,
        "showRectArroundTrafficSign": False
    }

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.video_file = None
        self.status = True
        self.cap = None
        self.isAvailable = True

        self.categories_color = np.array([[0, 0, 0]] + [obj["color"] for obj in CATEGORIES_COLORS.values()], dtype=np.uint8)


    def start_file(self, fname):
        self.video_file = os.path.join(VIDEO_PATH, fname)

        while self.isAvailable == False:
            pass

        self.start()

    def sendTo(self, evt, frame):
        # Creating and scaling QImage
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

        # Emit signal
        evt.emit(scaled_img)

    def changeOptions(self, name, checked):
        print(name, self.options[name], "-->", checked)
        self.options[name] = checked

        if name in ["showRoad", "showObjects", "showBackground"]:
            values = CATEGORIES_COLORS.values()
            self.categories_color = np.zeros((len(values) + 1, 3), dtype=np.uint8)
            for o, data in enumerate(values):
                i = o + 1
                if (i >= 1 and i <= 5 and self.options["showRoad"]) or (i >= 6 and i <= 13 and self.options["showObjects"]) or (i >= 14 and self.options["showBackground"]):
                    self.categories_color[i] = data["color"]

    def loadSegmentationModel(self, filename):
        self.segmentation_model = None
        self.segmentation_model_size = None

        try:
            self.segmentation_model = keras.models.load_model(filename, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
            self.segmentation_model_size = self.segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]
        except:
            print("[loadSegmentationModel] Une erreur est survenue lors de l'ouverture du h5")

    def loadTrafficSignRecognitionModel(self, filename):

        self.traffic_sign_recognition_model = None
        self.traffic_sign_recognition_model_size = None

        try:
            self.traffic_sign_recognition_model = keras.models.load_model(filename)
            self.traffic_sign_recognition_model_size = self.traffic_sign_recognition_model.get_layer(index=0).input_shape[0][1:-1][::-1]
        except Exception as e:
            print("[loadTrafficSignRecognitionModel] Une erreur est survenue lors de l'ouverture du h5")
            print(e)

    def run(self):

        global CATEGORIES_COLORS

        if self.video_file is not None:

            self.ThreadActive = True

            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_file)

            previous_frame = np.zeros((TIME_PERSISTANT, ) + self.segmentation_model.layers[-1].output_shape[1:])

            prev_frame_time = 0
            new_frame_time = 0

            while(self.ThreadActive and self.cap.isOpened()):

                self.isAvailable = False

                ret, frame = self.cap.read()
                new_frame_time = time.time()

                if not ret:
                    continue

                img_resized = cv2.resize(frame, self.segmentation_model_size, interpolation=cv2.INTER_AREA)

                result_segmentation = self.segmentation_model.predict(np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) / 255., axis=0))[0]
                result_segmentation_with_temp = result_segmentation

                if self.options["useTimeConsistency"]:
                    result_segmentation_with_temp = result_segmentation + previous_frame[0] + 0.5 * np.sum(previous_frame[:1])

                # Argmax
                argmax_result_segmentation = argmax(result_segmentation_with_temp, axis=-1)

                # Index --> Couleur
                argmax_result_segmentation = np.expand_dims(argmax_result_segmentation, axis=-1)
                segmentation = np.squeeze(np.take(self.categories_color, argmax_result_segmentation, axis=0))

                # En cas de détection de "Traffic Sign", on dessine une box autour
                if self.options["showRectArroundTrafficSign"]:
                    contours, _ = cv2.findContours(np.array(argmax_result_segmentation == 7, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        if w > 10 and h > 10 and w * h > 200:
                            x = x - BOUNDING_BOX_PADDING
                            y = y - BOUNDING_BOX_PADDING
                            w = w + BOUNDING_BOX_PADDING * 2
                            h = h + BOUNDING_BOX_PADDING * 2

                            x = x if x >= 0 else 0
                            y = y if y >= 0 else 0
                            w = w if w <= self.segmentation_model_size[0] else self.segmentation_model_size[0]
                            h = h if h <= self.segmentation_model_size[1] else self.segmentation_model_size[1]

                            cv2.rectangle(segmentation, (x, y), (x + w, y + h), (0, 255, 0), 1)

                            if self.traffic_sign_recognition_model is not None:
                                test_sign = cv2.resize(img_resized[y:y + h, x:x + w], self.traffic_sign_recognition_model_size, interpolation=cv2.INTER_AREA)
                                test_sign = np.array([test_sign / 255.])
                                result_traffic = self.traffic_sign_recognition_model.predict(test_sign)[0]
                                max_index_col = np.argmax(result_traffic, axis=0)
                                proba = result_traffic[max_index_col]
                                if proba > 0.85:
                                    try:
                                        img_resized = copyTrafficSign(img_resized, TRAFFIC_SIGN_DATASET_IMAGE[max_index_col], x, y)
                                    except Exception as err:
                                        print(err)

                # On redimenssione les résultats pour les afficher correctement
                if self.segmentation_model_size != (640, 480):
                    img_resized = cv2.resize(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_AREA)
                    segmentation = cv2.resize(segmentation, (640, 480), interpolation=cv2.INTER_AREA)

                # On tourne les résulats précédents et on sauvegarde le resultat
                previous_frame = np.roll(previous_frame, 1)
                previous_frame[0] = result_segmentation

                # On calcule le temps nécéssaire
                fps = 1 // (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # On envoie les données
                self.sendTo(self.EVT_ROAD_IMAGE, img_resized)
                self.sendTo(self.EVT_SEGMENTATION_IMAGE, segmentation)
                self.EVT_FPS.emit(fps)

            self.cap.release()
            self.isAvailable = True

    def stop(self):
        self.ThreadActive = False

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Titre
        self.setWindowTitle("Road Segmentation")

        # Thread in charge of updating the image
        self.thread = Thread(self)
        self.thread.EVT_ROAD_IMAGE.connect(self.setRoadImage)
        self.thread.EVT_SEGMENTATION_IMAGE.connect(self.setSegmentationImage)
        self.thread.EVT_FPS.connect(self.setFPS)

        # MODEL CHOOSER LAYOUT
        self.model_chooser_layout = QGroupBox("Model chooser")
        self.model_chooser_layout.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Segmentation UI
        segmentation_model_chooser_layout = QHBoxLayout()
        self.segmentation_model_chooser_input = QLineEdit()
        self.segmentation_model_chooser_button = QPushButton("...")

        segmentation_model_chooser_layout.addWidget(QLabel("Segmentation :"), 10)
        segmentation_model_chooser_layout.addWidget(self.segmentation_model_chooser_input, 50)
        segmentation_model_chooser_layout.addWidget(self.segmentation_model_chooser_button)

        # Traffic Sign Recognition UI
        traffic_sign_model_chooser_layout = QHBoxLayout()
        self.traffic_sign_model_chooser_input = QLineEdit()
        self.traffic_sign_model_chooser_button = QPushButton("...")

        traffic_sign_model_chooser_layout.addWidget(QLabel("Traffic Sign Recognition :"), 10)
        traffic_sign_model_chooser_layout.addWidget(self.traffic_sign_model_chooser_input, 50)
        traffic_sign_model_chooser_layout.addWidget(self.traffic_sign_model_chooser_button)

        # Model UI def
        model_chooser_layout = QHBoxLayout()
        model_chooser_layout.addLayout(segmentation_model_chooser_layout)
        model_chooser_layout.addLayout(traffic_sign_model_chooser_layout)
        self.model_chooser_layout.setLayout(model_chooser_layout)

        # IMAGE RESULT
        self.video_layout_model = QGroupBox("Result")
        self.video_layout_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        result_layout = QVBoxLayout()

        image_layout = QHBoxLayout()
        self.image_seg = QLabel(self)
        self.image_road = QLabel(self)
        self.image_seg.setFixedSize(640, 480)
        self.image_road.setFixedSize(640, 480)

        self.image_seg.setObjectName("image_seg")
        self.image_seg.setStyleSheet("#image_seg {background-color: black;}")

        self.image_road.setObjectName("image_road")
        self.image_road.setStyleSheet("#image_road {background-color: black;}")

        image_layout.addWidget(self.image_road)
        image_layout.addWidget(self.image_seg)

        # LABELS
        label_scroll = QScrollArea()
        label_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        grid = QGridLayout()

        for id_label in CATEGORIES_COLORS:

            label = CATEGORIES_COLORS[id_label]
            color = "rgb(" + str(label["color"][0]) + "," + str(label["color"][1]) + "," + str(label["color"][2]) + ")"

            testWidget = QFrame()
            testWidget.setFixedSize(50, 25)
            testWidget.setObjectName("myWidget")
            testWidget.setStyleSheet("#myWidget {background-color:" + color + ";}")

            grid.addWidget(testWidget, id_label, 0)
            grid.addWidget(QLabel(label["name"]), id_label, 1)

        label_widget = QWidget()
        label_widget.setLayout(grid)
        label_scroll.setWidget(label_widget)
        label_scroll.setFixedWidth(200)

        image_layout.addWidget(label_scroll)
        result_layout.addLayout(image_layout)

        # Showing FPS
        self.fps_label = QLabel("FPS :")
        result_layout.addWidget(self.fps_label)

        self.video_layout_model.setLayout(result_layout)

        # VIDEO FILE CHOOSER
        self.group_model = QGroupBox("Video file")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()
        self.combobox = QComboBox()
        for video_filename in os.listdir(VIDEO_PATH):
            self.combobox.addItem(video_filename)

        model_layout.addWidget(QLabel("Video :"), 10)
        model_layout.addWidget(self.combobox, 75)
        self.group_model.setLayout(model_layout)

        # BUTTONS
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.start_button)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.group_model, 1)
        control_layout.addLayout(buttons_layout, 1)

        # OPTIONS
        option_layout = QGroupBox("Options")
        option_layout.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        option_layout_in = QHBoxLayout()

        self.showRoad = QCheckBox("Show road")
        self.showRoad.setChecked(True)
        self.showObjects = QCheckBox("Show objects")
        self.showObjects.setChecked(True)
        self.showBackground = QCheckBox("Show background")
        self.showBackground.setChecked(True)
        self.useTimeConsistency = QCheckBox("Time consistent result")
        self.useTimeConsistency.setChecked(True)
        self.showRectArroundTrafficSign = QCheckBox("Show traffic signs")

        option_layout_in.addWidget(self.showRoad)
        option_layout_in.addWidget(self.showObjects)
        option_layout_in.addWidget(self.showBackground)
        option_layout_in.addWidget(self.useTimeConsistency)
        option_layout_in.addWidget(self.showRectArroundTrafficSign)
        option_layout.setLayout(option_layout_in)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.model_chooser_layout)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(option_layout)
        main_layout.addWidget(self.video_layout_model)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Connections
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.combobox.currentTextChanged.connect(self.set_video)
        self.segmentation_model_chooser_input.returnPressed.connect(self.segmentation_loadModel_Input)
        self.segmentation_model_chooser_button.clicked.connect(self.segmentation_loadModel_Button)
        self.traffic_sign_model_chooser_input.returnPressed.connect(self.traffic_sign_loadModel_Input)
        self.traffic_sign_model_chooser_button.clicked.connect(self.traffic_sign_loadModel_Button)

        self.showRoad.clicked.connect(self.showRoad_LISTENER)
        self.showObjects.clicked.connect(self.showObjects_LISTENER)
        self.showBackground.clicked.connect(self.showBackground_LISTENER)
        self.useTimeConsistency.clicked.connect(self.useTimeConsistency_LISTENER)
        self.showRectArroundTrafficSign.clicked.connect(self.showRectArroundTrafficSign_LISTENER)

    @Slot()
    def set_video(self, filename):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.thread.start_file(filename)
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

    @Slot()
    def start(self):
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.thread.start_file(self.combobox.currentText())

    @Slot()
    def stop(self):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    @Slot(QImage)
    def setRoadImage(self, image):
        self.image_road.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def setSegmentationImage(self, image):
        self.image_seg.setPixmap(QPixmap.fromImage(image))

    @Slot(int)
    def setFPS(self, fps):
        self.fps_label.setText("FPS : " + str(fps))

    def segmentation_loadModel_Input(self):
        fileName = self.segmentation_model_chooser_input.text()
        self.thread.loadSegmentationModel(fileName)

    def segmentation_loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.segmentation_model_chooser_input.setText(fileName[0])
        self.thread.loadSegmentationModel(fileName[0])

    def traffic_sign_loadModel_Input(self):
        fileName = self.traffic_sign_model_chooser_input.text()
        self.thread.loadTrafficSignRecognitionModel(fileName)

    def traffic_sign_loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.traffic_sign_model_chooser_input.setText(fileName[0])
        self.thread.loadTrafficSignRecognitionModel(fileName[0])

    def showRoad_LISTENER(self, checked):
        self.thread.changeOptions("showRoad", checked)

    def showObjects_LISTENER(self, checked):
        self.thread.changeOptions("showObjects", checked)

    def showBackground_LISTENER(self, checked):
        self.thread.changeOptions("showBackground", checked)

    def useTimeConsistency_LISTENER(self, checked):
        self.thread.changeOptions("useTimeConsistency", checked)

    def showRectArroundTrafficSign_LISTENER(self, checked):
        self.thread.changeOptions("showRectArroundTrafficSign", checked)

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":

    sys.excepthook = except_hook

    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
