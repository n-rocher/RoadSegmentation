import os
import sys
import cv2
import signal
import numpy as np

from tensorflow import keras, argmax

from utils.argmaxMeanIOU import ArgmaxMeanIOU
from utils.dataset import CATEGORIES_COLORS

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

IMG_SIZE = (720, 480)
MORPHOLOGY_KERNEL = np.ones((2, 2), 'uint8')

VIDEO_PATH = ""

class Thread(QThread):
    EVT_ROAD_IMAGE = Signal(QImage)
    EVT_SEGMENTATION_IMAGE = Signal(QImage)

    segmentation_model = None
    segmentation_model_size = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.video_file = None
        self.status = True
        self.cap = None
        self.isAvailable = True

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

    def loadSegmentationModel(self, filename):
        self.segmentation_model = None
        self.segmentation_model_size = None

        try:
            self.segmentation_model = keras.models.load_model(filename, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
            self.segmentation_model_size = self.segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]
        except:
            print("[loadSegmentationModel] Une erreur est survenue lors de l'ouverture du h5")

    def run(self):

        global CATEGORIES_COLORS

        if self.video_file is not None:

            self.ThreadActive = True

            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_file)

            while(self.ThreadActive and self.cap.isOpened()):
             
                self.isAvailable = False

                ret, frame = self.cap.read()
                if not ret:
                    continue

              
                img_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_AREA)
                self.sendTo(self.EVT_ROAD_IMAGE, img_resized)


                if self.segmentation_model:
                    img_resized = cv2.resize(frame, self.segmentation_model_size, interpolation=cv2.INTER_AREA)
                    
                    result_segmentation = self.segmentation_model.predict(np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR) / 255., axis=0))[0]

                    # Argmax
                    result_segmentation = argmax(result_segmentation, axis=-1)

                    result_segmentation = cv2.morphologyEx(np.array(result_segmentation, dtype=np.uint8), cv2.MORPH_OPEN, MORPHOLOGY_KERNEL)

                    segmentation = np.zeros(result_segmentation.shape + (3,), dtype=np.uint8)
                    for categorie in CATEGORIES_COLORS.keys():
                        segmentation[result_segmentation == categorie] = CATEGORIES_COLORS[categorie]["color"]

                    if self.segmentation_model_size != (640, 480):
                        img_resized = cv2.resize(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_AREA)
                        segmentation = cv2.resize(segmentation, (640, 480), interpolation=cv2.INTER_AREA)

                    self.sendTo(self.EVT_SEGMENTATION_IMAGE, segmentation)


            self.cap.release()
            self.isAvailable = True

    def stop(self):
        self.ThreadActive = False

class SegmentationWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Titre
        self.setWindowTitle("Road Segmentation")

        # Thread in charge of updating the image
        self.thread = Thread(self)
        self.thread.EVT_ROAD_IMAGE.connect(self.setRoadImage)
        self.thread.EVT_SEGMENTATION_IMAGE.connect(self.setSegmentationImage)

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

        # Model UI def
        model_chooser_layout = QHBoxLayout()
        model_chooser_layout.addLayout(segmentation_model_chooser_layout)
        self.model_chooser_layout.setLayout(model_chooser_layout)

        # IMAGE RESULT
        self.video_layout_model = QGroupBox("Result")
        self.video_layout_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        image_layout = QHBoxLayout()
        self.image_seg = QLabel(self)
        self.image_road = QLabel(self)
        self.image_seg.setFixedSize(640, 480)
        self.image_road.setFixedSize(640, 480)
        image_layout.addWidget(self.image_road)
        image_layout.addWidget(self.image_seg)
        self.video_layout_model.setLayout(image_layout)

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

        # Main layout
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(self.model_chooser_layout)
        layout.addLayout(control_layout)
        layout.addWidget(self.video_layout_model)

        main_layout.addLayout(layout)

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

        main_layout.addWidget(label_scroll)

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

    def segmentation_loadModel_Input(self):
        fileName = self.segmentation_model_chooser_input.text()
        self.thread.loadSegmentationModel(fileName)

    def segmentation_loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.segmentation_model_chooser_input.setText(fileName[0])
        self.thread.loadSegmentationModel(fileName[0])





if __name__ == "__main__":

    if len(sys.argv) == 2:

        VIDEO_PATH = sys.argv[1]

        app = QApplication()
        seg_win = SegmentationWindow()

        def sigint_handler(*args):
            seg_win.thread.stop()
            QApplication.quit()

        signal.signal(signal.SIGINT, sigint_handler)

        seg_win.show()

        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)

        sys.exit(app.exec())
    else:
        print("Error: Add the video folder argument")