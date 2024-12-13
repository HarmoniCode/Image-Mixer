import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFrame, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ImageData(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.image = None
        self.magnitude_sepctrum = None
        self.phase_sepctrum = None
        self.transformed = None

        self.label = QLabel("Image", self.image)
        self.label.setMaximumWidth(200)
        self.label.setMinimumWidth(150)
        self.label.setMaximumHeight(200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("image_label")

        self.component_canvas = FigureCanvas(Figure(figsize=(2, 2)))
        self.component_canvas.setFixedSize(200, 200)
        self.ax = self.component_canvas.figure.add_subplot(111)


        self.combo_box = QComboBox()
        self.combo_box.addItem("Magnitude")
        self.combo_box.addItem("Phase")
        self.combo_box.currentIndexChanged.connect(self.update_component_display)


        self.label.mouseDoubleClickEvent = lambda event: self.load_image()

        H_layout = QHBoxLayout()
        H_layout.addWidget(self.label)
        H_layout.addWidget(self.component_canvas)

        self.layout.addLayout(H_layout)
        self.layout.addWidget(self.combo_box)
        self.layout.addWidget(self.combo_box)

        self.setLayout(self.layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(self.image, (300, 300))
            self.calculate_frequency_components()
            self.display_image(self.label)
            self.update_component_display()

    def calculate_frequency_components(self):
        if self.image is not None:
            self.transformed = np.fft.fft2(self.image)
            self.magnitude_sepctrum = np.abs(self.transformed)
            self.phase_sepctrum = np.angle(self.transformed)



    def display_image(self, label):
        if self.image is not None:
            height, width = self.image.shape
            bytes_per_line = width
            image_bytes = self.image.tobytes()
            qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))  

    def update_component_display(self):

        fshift = np.fft.fftshift(self.transformed)
        if self.image is not None:
            if self.combo_box.currentText() == "Magnitude":
                
                component = 20 * np.log(np.abs(fshift) + 1e-5)
            else:
                component = np.angle(fshift)

            self.ax.clear()
            self.ax.imshow(component, cmap='gray')
            self.ax.axis('off')
            self.component_canvas.draw()


class outputPort(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.label = QLabel("Reconstructed Image")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setMaximumWidth(300)
        self.label.setMinimumWidth(300)
        self.label.setMaximumHeight(300)

        self.layout.addWidget(self.label)

        self.process_button=QPushButton("Process and Reconstruct")
        self.layout.addWidget(self.process_button)

        self.control_frame = QFrame()
        self.control_layout = QVBoxLayout()
        self.control_frame.setLayout(self.control_layout)

        self.weight_sliders = []
        self.combo_boxes = []

        for i in range(4):
            self.weight_slider = QSlider(Qt.Orientation.Horizontal)
            self.weight_slider.setRange(0, 100)
            self.weight_slider.setValue(0)
            self.weight_slider.setTickInterval(1)
            self.weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.weight_slider.setFixedWidth(200)
            self.weight_sliders.append(self.weight_slider)

            self.combo_box = QComboBox()
            self.combo_box.addItem("Magnitude")
            self.combo_box.addItem("Phase")

            self.combo_boxes.append(self.combo_box)

            self.control_layout.addWidget(self.combo_box)
            self.control_layout.addWidget(self.weight_slider)        

        self.layout.addWidget(self.control_frame)
        self.setLayout(self.layout)


class ImageReconstructionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Frequency Reconstruction with Weight Sliders")
        self.setGeometry(200, 200, 1000, 600)

        self.right_frame = QFrame()
        self.right_frame.setMaximumWidth(400)
        self.right_frame.setMinimumWidth(400)

        self.right_layout = QVBoxLayout()
        self.right_frame.setLayout(self.right_layout)

        self.left_frame = QFrame()
        self.left_layout = QVBoxLayout()
        self.left_frame.setLayout(self.left_layout)

        self.middle_frame = QFrame()
        self.middle_layout = QVBoxLayout()
        self.middle_frame.setLayout(self.middle_layout)

        self.controll_frame = QFrame()
        self.controll_layout = QVBoxLayout()
        self.controll_frame.setLayout(self.controll_layout)

        self.layout = QHBoxLayout()

        # Variables to store image objects
        self.image_1 = ImageData()
        self.image_2 = ImageData()
        self.image_3 = ImageData()
        self.image_4 = ImageData()

        self.output_port_1 = outputPort()
        self.output_port_2 = outputPort()

        self.process_button = QPushButton("Process and Reconstruct", self)
        images=[self.image_1, self.image_2, self.image_3, self.image_4]


        # Horizontal layout for displaying the images
        self.image_layout = QHBoxLayout()

        self.reconstructed_label = QLabel(self)

        H_layout_1 = QHBoxLayout()
        H_layout_2 = QHBoxLayout()

        H_layout_1.addWidget(self.image_1)
        H_layout_1.addWidget(self.image_2)

        H_layout_2.addWidget(self.image_3)
        H_layout_2.addWidget(self.image_4)

        self.middle_layout.addLayout(H_layout_1)
        self.middle_layout.addLayout(H_layout_2)

        # self.right_layout.addWidget(self.reconstructed_label)
        # self.right_layout.addWidget(self.process_button)
        # self.right_layout.addWidget(self.controll_frame)

        self.right_layout.addWidget(self.output_port_1)
        self.left_layout.addWidget(self.output_port_2)

        

        self.layout.addWidget(self.left_frame)
        self.layout.addWidget(self.middle_frame)
        self.layout.addWidget(self.right_frame)

        # Buttons for loading images and processing



        # Connect buttons to functions
        self.output_port_1.process_button.clicked.connect(lambda: self.process_images(self.output_port_1))
        self.output_port_2.process_button.clicked.connect(lambda: self.process_images(self.output_port_2))


        self.setLayout(self.layout)

    def process_images(self, output_port):
        images = [self.image_1, self.image_2, self.image_3, self.image_4]

        for image in images:
            if image.image is not None:
                image.calculate_frequency_components()

        if all(image.image is not None for image in images):
            magnitude_components = np.zeros_like(images[0].magnitude_sepctrum)
            phase_components = np.zeros_like(images[0].phase_sepctrum)

            for i in range(4):
                if output_port.combo_boxes[i].currentText() == "Magnitude":
                    magnitude_components += (output_port.weight_sliders[i].value() / 100.0) * images[i].magnitude_sepctrum
                elif output_port.combo_boxes[i].currentText() == "Phase":
                    phase_components += (output_port.weight_sliders[i].value() / 100.0) * images[i].phase_sepctrum

            total_magnitude_weight = sum(output_port.weight_sliders[i].value() for i in range(4) if output_port.combo_boxes[i].currentText() == "Magnitude") / 100.0
            total_phase_weight = sum(output_port.weight_sliders[i].value() for i in range(4) if output_port.combo_boxes[i].currentText() == "Phase") / 100.0

            if total_magnitude_weight > 0:
                magnitude_components /= total_magnitude_weight
            if total_phase_weight > 0:
                phase_components /= total_phase_weight

            reconstructed_f = magnitude_components * np.exp(1j * phase_components)
            reconstructed_image = np.abs(np.fft.ifft2(reconstructed_f))
            reconstructed_image = np.uint8(np.clip(reconstructed_image, 0, 255))

            height, width = reconstructed_image.shape
            bytes_per_line = width
            image_bytes = reconstructed_image.tobytes()
            qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            output_port.label.setPixmap(pixmap.scaled(output_port.label.width(), output_port.label.height(), Qt.KeepAspectRatio))  # Resize to fit label

        else:
            print("Please load images.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageReconstructionApp()
    window.show()
    sys.exit(app.exec_())