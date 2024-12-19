import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QFrame, QComboBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

class ImageData(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.image = None
        self.magnitude_spectrum = None
        self.phase_spectrum = None
        self.transformed = None

        self.label = QLabel("Image", self.image)
        self.label.setMaximumWidth(250)
        self.label.setMinimumWidth(250)
        self.label.setMaximumHeight(250)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("image_label")

        self.component_canvas = FigureCanvas(Figure(figsize=(2, 2)))
        self.component_canvas.setFixedSize(250, 250)
        self.ax = self.component_canvas.figure.add_subplot(111)
        self.ax.axis('off') 

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
        self.rectangle_selector = RectangleSelector(
            self.ax,
            onselect=None,
            interactive=True,
            useblit=True,  
            drag_from_anywhere=True,
            spancoords='pixels',
        )
        self.rectangle_selector.set_active(True)

    def load_image(self, file_path=None):
        if file_path is None:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")

        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(self.image, (250, 250))
            self.display_image(self.image)
            self.update_component_display()

    def calculate_frequency_components(self):
        if self.image is not None:
            self.transformed = np.fft.fftshift(np.fft.fft2(self.image))
            self.magnitude_spectrum = np.abs(self.transformed)
            self.phase_spectrum = np.angle(self.transformed)



    def display_image(self, image):
        if image is not None:
            height, width = image.shape
            bytes_per_line = width
            image_bytes = image.tobytes()
            qimage = QImage(image_bytes, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            self.label.setPixmap(pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio))
            self.ax.imshow(image, cmap='gray')
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(height, 0)
            self.component_canvas.draw()
            
    def update_component_display(self):
        if self.image is None:
            return

        if self.combo_box.currentText() == "Magnitude":
            component = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(self.image))) + 1e-5)
        else:
            component = np.angle(np.fft.fftshift(np.fft.fft2(self.image)))

        self.ax.clear()
        self.ax.imshow(component, cmap='gray')
        self.ax.set_xlim(0, self.image.shape[1])
        self.ax.set_ylim(self.image.shape[0], 0)
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
        self.right_frame.setMaximumWidth(350)
        self.right_frame.setMinimumWidth(350)

        self.right_layout = QVBoxLayout()
        self.right_frame.setLayout(self.right_layout)

        self.left_frame = QFrame()
        self.left_frame.setMaximumWidth(350)
        self.left_frame.setMinimumWidth(350)
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
        # Buttons for loading images and processing
        self.reset_button = QPushButton("Reset Coordinates")
        self.reset_button.clicked.connect(self.reset_coordinates)
        self.middle_layout.addWidget(self.reset_button)
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

        self.image_1.rectangle_selector.onselect = self.on_select
        self.image_2.rectangle_selector.onselect = self.on_select
        self.image_3.rectangle_selector.onselect = self.on_select
        self.image_4.rectangle_selector.onselect = self.on_select
        
        self.selected_region = [0,300,0,300]
        
        self.load_initial_images()

    import matplotlib.pyplot as plt

    def reset_coordinates(self):
        self.selected_region = [0, 250, 0, 250]
        for image in [self.image_1, self.image_2, self.image_3, self.image_4]:
            image.rectangle_selector.extents = (0, 250, 0, 250)
            image.rectangle_selector.update()
    def load_initial_images(self):
        image_paths = [
            'data/image1.jpg',
            'data/image2.jpg',
            'data/image3.jpg',
            'data/image4.jpg'
        ]
        images = [self.image_1, self.image_2, self.image_3, self.image_4]

        fig, axes = plt.subplots(4, 3, figsize=(15, 20))  # Updated layout for 3 columns per row

        for i, (image, path) in enumerate(zip(images, image_paths)):
            image.load_image(path)
            image.rectangle_selector.extents = (0, 300, 0, 300)
            image.rectangle_selector.update()
            image.calculate_frequency_components()  # Calculate frequency components
            image.update_component_display()

            # Get Fourier Transform and its dimensions
            transformed = np.fft.fftshift(image.transformed)
            height, width = transformed.shape

            # Compute magnitude spectrum
            magnitude_spectrum = 20 * np.log(np.abs(transformed) + 1e-5)
            axes[i, 0].imshow(magnitude_spectrum, cmap='viridis')
            axes[i, 0].set_title(f'Magnitude Spectrum {i + 1}')
            axes[i, 0].axis('off')

            # Compute phase spectrum
            phase_spectrum = np.angle(transformed)
            axes[i, 1].imshow(phase_spectrum, cmap='twilight')
            axes[i, 1].set_title(f'Phase Spectrum {i + 1}')
            axes[i, 1].axis('off')

            # Radial frequency spectrum calculation
            y, x = np.indices((height, width))
            center = (height // 2, width // 2)  # Center of the FFT
            radial_distances = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

            # Sort magnitude values into radial bins
            radial_distances = radial_distances.astype(int)
            max_radius = radial_distances.max()
            radial_profile = np.bincount(radial_distances.ravel(), weights=magnitude_spectrum.ravel()) / np.bincount(
                radial_distances.ravel())

            # Frequency axis in Hz (truncate to match radial_profile)
            freq = np.fft.fftfreq(max(width, height))[:len(radial_profile)]

            # Plot frequency vs amplitude
            axes[i, 2].plot(freq, radial_profile, color='blue')
            axes[i, 2].set_title(f'Frequency Spectrum {i + 1}')
            axes[i, 2].set_xlabel('Frequency (Hz)')
            axes[i, 2].set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    def on_select(self, eclick, erelease):
            x0, y0 = round(eclick.xdata), round(eclick.ydata)
            x1, y1 = round(erelease.xdata), round(erelease.ydata)

            print(f"Normalized coordinates (x0, y0): ({x0}, {y0})")
            print(f"Normalized coordinates (x1, y1): ({x1}, {y1})")

            if x0 == x1 or y0 == y1:
                return
            self.selected_region = [y0, y1, x0, x1]

            for image in [self.image_1, self.image_2, self.image_3, self.image_4]:
                if image.magnitude_spectrum is not None:
                                       
                    image.rectangle_selector.extents = (x0, x1, y0, y1)
                    image.rectangle_selector.update()
                    
                    
    def process_images(self, output_port):
        images = [self.image_1, self.image_2, self.image_3, self.image_4]

        for image in images:
            if image.image is not None:
                image.calculate_frequency_components()

        if all(image.image is not None for image in images):
            magnitude_components = np.zeros_like(images[0].magnitude_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]])
            phase_components = np.zeros_like(images[0].phase_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]])


            for i in range(4):
                if output_port.combo_boxes[i].currentText() == "Magnitude":
                    magnitude_components += (output_port.weight_sliders[i].value() / 100.0) * images[i].magnitude_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]]
                    
                    
                elif output_port.combo_boxes[i].currentText() == "Phase":
                    phase_components += (output_port.weight_sliders[i].value() / 100.0) * images[i].phase_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]]
                   
                print(images[0].magnitude_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]])
                
                print(f'mag components  {magnitude_components}')
                print(f'phase components {phase_components}')
                print(images[1].phase_spectrum[
                    self.selected_region[0] : self.selected_region[1],
                    self.selected_region[2] : self.selected_region[3]])

            total_magnitude_weight = sum(output_port.weight_sliders[i].value() for i in range(4) if output_port.combo_boxes[i].currentText() == "Magnitude") / 100.0
            total_phase_weight = sum(output_port.weight_sliders[i].value() for i in range(4) if output_port.combo_boxes[i].currentText() == "Phase") / 100.0
            print(f"sum mag {total_magnitude_weight}")
            print(f"sum phase {total_phase_weight}")

            if total_magnitude_weight > 0:
                magnitude_components /= total_magnitude_weight
            if total_phase_weight > 0:
                phase_components /= total_phase_weight
            print(f"total mag {magnitude_components}")
            print(f"total phase {phase_components}")

            reconstructed_f = magnitude_components * np.exp(1j * phase_components)
            reconstructed_image = np.abs(np.fft.ifft2(reconstructed_f))
            print(f"reconstructed image before{reconstructed_image}")
            reconstructed_image = np.uint8(np.clip(reconstructed_image, 0, 255))
            print(f"reconstructed image after{reconstructed_image}")

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