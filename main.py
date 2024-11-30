import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt


class ImageGroup(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent) 
        self.layout = QVBoxLayout()

        self.label = QLabel("Image", self)
        self.label.setMaximumWidth(400)
        self.label.setMinimumWidth(300)
        self.label.setMaximumHeight(400)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label.setObjectName("image_label")  

        self.magnitude_canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.phase_canvas = FigureCanvas(Figure(figsize=(4, 4)))

        self.layout.addWidget(self.label)

        self.h_layout = QHBoxLayout()
        self.h_layout.addLayout(self.layout)
        self.h_layout.addWidget(self.magnitude_canvas)
        self.h_layout.addWidget(self.phase_canvas)

        self.setLayout(self.h_layout)


class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image FFT Viewer")
        self.setGeometry(100, 100, 1200, 400)

        h_layout1 = QHBoxLayout()
        h_layout2 = QHBoxLayout()

        self.image_group1 = ImageGroup(self)
        self.image_group2 = ImageGroup(self)
        self.image_group3 = ImageGroup(self)
        self.image_group4 = ImageGroup(self)

        h_layout1.addWidget(self.image_group1)
        h_layout1.addWidget(self.image_group2)

        h_layout2.addWidget(self.image_group3)
        h_layout2.addWidget(self.image_group4)


        self.image_group1.label.mouseDoubleClickEvent = lambda event: self.load_image(
            self.image_group1.label, self.image_group1.magnitude_canvas, self.image_group1.phase_canvas
        )
        self.image_group2.label.mouseDoubleClickEvent = lambda event: self.load_image(
            self.image_group2.label, self.image_group2.magnitude_canvas, self.image_group2.phase_canvas
        )
        self.image_group3.label.mouseDoubleClickEvent = lambda event: self.load_image(
            self.image_group3.label, self.image_group3.magnitude_canvas, self.image_group3.phase_canvas
        )
        self.image_group4.label.mouseDoubleClickEvent = lambda event: self.load_image(
            self.image_group4.label, self.image_group4.magnitude_canvas, self.image_group4.phase_canvas
        )

        main_layout = QVBoxLayout()
        main_layout.addLayout(h_layout1)
        main_layout.addLayout(h_layout2)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self, label, mag_canvas, phase_canvas):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.show_image(image, label)

            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            phase_spectrum = np.angle(fshift)

            self.show_fft(magnitude_spectrum, mag_canvas)
            self.show_fft(phase_spectrum, phase_canvas)

    def show_image(self, image, label):
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def show_fft(self, spectrum, canvas):
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.imshow(spectrum, cmap='gray')
        ax.axis('off')
        canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    ex.show()
    sys.exit(app.exec_())