import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox,
    QSlider, QRadioButton, QPushButton,QButtonGroup

)
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

from matplotlib.widgets import RectangleSelector


class ImageGroup(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout()
        self.image_slider_layout = QVBoxLayout()
        self.label = QLabel("Image", self)
        self.label.setMaximumWidth(200)
        self.label.setMinimumWidth(150)
        self.label.setMaximumHeight(200)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("image_label")

        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.weight_slider.setRange(0, 100)
        self.weight_slider.setValue(0)
        self.weight_slider.setTickInterval(1)
        self.weight_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.weight_slider.setFixedWidth(200)

        self.canvas_group = QVBoxLayout()
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("Magnitude")
        self.combo_box.addItem("Phase")
        self.combo_box.addItem("Real")
        self.combo_box.addItem("Imaginary")
        self.combo_box.currentIndexChanged.connect(self.update_canvas)

        self.canvas = FigureCanvas(Figure(figsize=(2, 2)))
        self.canvas.setFixedSize(200, 200)
        self.ax = self.canvas.figure.add_subplot(111)

        self.canvas_group.addWidget(self.combo_box)
        self.canvas_group.addWidget(self.canvas)

        self.image_slider_layout.addWidget(self.weight_slider)
        self.image_slider_layout.addWidget(self.label)
        self.layout.addLayout(self.image_slider_layout)
        self.layout.addLayout(self.canvas_group)

        self.setLayout(self.layout)
        self.magnitude_spectrum = None
        self.phase_spectrum = None
        
        self.brightness = 0  
        self.contrast = 1.0  
        self.original_image = None  
        self.start_pos = None        
        self.label.mousePressEvent = self.start_mouse_drag
        self.label.mouseMoveEvent = self.adjust_brightness_contrast
        
        # Rectangle selector
        self.rectangle_selector = RectangleSelector(
            self.ax,
            onselect=None,
            interactive=True,
            useblit=True,  
            drag_from_anywhere=True,
            use_data_coordinates=True,
            spancoords="pixels"
        )
        self.rectangle_selector.set_active(True)


    def update_canvas(self):
        '''Update the canvas with the selected spectrum'''
        if self.magnitude_spectrum is not None and self.phase_spectrum is not None:
            if self.combo_box.currentText() == "Magnitude":
                self.show_fft(self.magnitude_spectrum)
            else:
                self.show_fft(self.phase_spectrum)

    def show_fft(self, spectrum):
        '''Show the FFT spectrum on the canvas
        Args:
            spectrum (np.ndarray): FFT spectrum
        '''
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.imshow(spectrum, cmap='gray')
        ax.axis('off')
        self.canvas.draw()

    def start_mouse_drag(self, event):
        '''Start tracking the mouse drag'''
        self.start_pos = event.pos()

    def adjust_brightness_contrast(self, event):

        '''Adjust brightness and contrast based on mouse movement'''
        if self.original_image is None:
            return

        dx = event.pos().x() - self.start_pos.x()  
        dy = event.pos().y() - self.start_pos.y()  
        
        self.brightness = dy * 0.5  
        self.contrast = 1 + (dx * 0.01)  

        adjusted_image = self.apply_brightness_contrast(self.original_image, self.brightness, self.contrast)
        self.display_image(adjusted_image)
        
        f = np.fft.fft2(adjusted_image)
        fshift = np.fft.fftshift(f)
        self.magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
        self.phase_spectrum = np.angle(fshift)
        
        self.update_canvas()

    def apply_brightness_contrast(self, image, brightness, contrast):
        '''Apply brightness and contrast adjustments'''
        adjusted = np.clip(contrast * image + brightness, 0, 255).astype(np.uint8)
        return adjusted
    
    def display_image(self, image):
        '''Update the QLabel with the adjusted image'''
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
        self.label.setScaledContents(True)

    def set_image(self, image):
        '''Set the original image for the label'''
        self.original_image = image
        self.display_image(image)


class OutPort(QWidget):
    def __init__(self, parent=None, label=""):
        super().__init__(parent)
        self.layout = QVBoxLayout()

        self.label = QLabel(label, self)
        self.label.setMaximumWidth(600)
        self.label.setMinimumWidth(400)
        self.label.setMaximumHeight(400)
        self.label.setStyleSheet("border: 1px solid black;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("image_label")  
        
        self.radio_state = False
        self.radio = QRadioButton(label)
        self.radio.setChecked(True)
        self.layout.addWidget(self.radio)
        self.layout.addWidget(self.label)

        self.setLayout(self.layout)

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Image FFT Viewer")
        self.setGeometry(100, 100, 600, 800)  

        self.image_group1 = ImageGroup(self)
        self.image_group2 = ImageGroup(self)
        self.image_group3 = ImageGroup(self)
        self.image_group4 = ImageGroup(self)

        v_layout1 = QVBoxLayout()
        v_layout1.addWidget(self.image_group1)
        v_layout1.addWidget(self.image_group2)
        v_layout1.addWidget(self.image_group3)
        v_layout1.addWidget(self.image_group4)

        self.image_group1.label.mouseDoubleClickEvent = lambda event: self.load_image(self.image_group1)
        self.image_group2.label.mouseDoubleClickEvent = lambda event: self.load_image(self.image_group2)
        self.image_group3.label.mouseDoubleClickEvent = lambda event: self.load_image(self.image_group3)
        self.image_group4.label.mouseDoubleClickEvent = lambda event: self.load_image(self.image_group4)
     
        self.out_port1 = OutPort(self, "Port 1")
        self.out_port2 = OutPort(self, "Port 2")
        self.out_port1.radio.setChecked(True)
        v_layout2 = QVBoxLayout()
        v_layout2.addWidget(self.out_port1)
        v_layout2.addWidget(self.out_port2)
        
        mix_button = QPushButton("Mix")
        mix_button.clicked.connect(self.mix_images)

        mode_group = QButtonGroup()
        region_group = QButtonGroup()
        
        mode_radio1 = QRadioButton("Magnitude / Phase")
        mode_radio1.setChecked(True)
        mode_group.addButton(mode_radio1)
        
        mode_radio2 = QRadioButton("Real / Imaginary")
        mode_radio2.setChecked(False)
        mode_group.addButton(mode_radio2)
        v_layout2.addWidget(mode_radio1)
        v_layout2.addWidget(mode_radio2)
        
        inside_region_radio = QRadioButton("Region inside")
        inside_region_radio.setChecked(True)
        region_group.addButton(inside_region_radio)
        
        outside_region_radio = QRadioButton("Region outside")
        outside_region_radio.setChecked(False)
        region_group.addButton(outside_region_radio)
       
        v_layout2.addWidget(inside_region_radio)
        v_layout2.addWidget(outside_region_radio)

        v_layout2.addWidget(mix_button)
        
        main_layout = QHBoxLayout()
        main_layout.addLayout(v_layout1)
        main_layout.addLayout(v_layout2)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.mixed_magnitude = None
        self.mixed_phase = None
        self.mixed_real = None
        self.mixed_imaginary = None

        self.image_group1.rectangle_selector.onselect = self.on_select
        self.image_group2.rectangle_selector.onselect = self.on_select
        self.image_group3.rectangle_selector.onselect = self.on_select
        self.image_group4.rectangle_selector.onselect = self.on_select

    def on_select(self, eclick, erelease):
        print('onselect')
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata

        print(f"Normalized coordinates (x0, y0): ({x0}, {y0})")
        print(f"Normalized coordinates (x1, y1): ({x1}, {y1})")

        if x0 == x1 or y0 == y1:
            print("Selected region has zero width or height. Selection failed.")
            return

        for i, image_group in enumerate([self.image_group1, self.image_group2, self.image_group3, self.image_group4]):
            if image_group.magnitude_spectrum is not None:
                height, width = image_group.magnitude_spectrum.shape

                x_min = int(x0 * width)
                x_max = int(x1 * width)
                y_min = int(y0 * height)
                y_max = int(y1 * height)

                x_min = max(0, min(x_min, width - 1))
                x_max = max(0, min(x_max, width - 1))
                y_min = max(0, min(y_min, height - 1))
                y_max = max(0, min(y_max, height - 1))

                print(f"Image Group {i+1} - Mapped coordinates (x_min, y_min): ({x_min}, {y_min})")
                print(f"Image Group {i+1} - Mapped coordinates (x_max, y_max): ({x_max}, {y_max})")

                selected_region = image_group.magnitude_spectrum[y_min:y_max, x_min:x_max]

                print(f"Image Group {i+1} - Selected region shape: {selected_region.shape}")            
            else:
                print(f"Image Group {i+1} - Magnitude spectrum is not available.")

        for i, image_group in enumerate([self.image_group1, self.image_group2, self.image_group3, self.image_group4]):
            image_group.rectangle_selector.extents = (x0, x1, y0, y1)
            image_group.rectangle_selector.update()

    def load_image(self, image_group):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_group.set_image(image)  

            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            image_group.magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
            image_group.phase_spectrum = np.angle(fshift)

            image_group.update_canvas()

    def show_image(self, image, label):
        '''Show the image on the label
        Args:
            image (np.ndarray): Image
            label (QLabel): QLabel object
        '''
        h, w = image.shape
        qimage = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        
    def get_weights(self):
        weight_1  = self.image_group1.weight_slider.value()
        weight_2 = self.image_group2.weight_slider.value()
        weight_3 = self.image_group3.weight_slider.value()
        weight_4 = self.image_group4.weight_slider.value()
        return weight_1, weight_2, weight_3, weight_4, sum(weight_1, weight_2, weight_3, weight_4)
    
   
    
    
    
    
    def mix_images(self,mode):
        self.mixed_magnitude, self.mixed_phase, self.mixed_real, self.mixed_imaginary = 0,0,0,0
        weight_1, weight_2, weight_3, weight_4, weights_sum = self.get_weights()
        if mode == "Mag and Phase":
            if  self.image_group1.combo_box.currentText() == "Magnitude":
                self.mixed_magnitude+= (weight_1 / weights_sum) * self.image_group1.magnitude_spectrum
            else:
                self.mixed_phase+= (weight_1 / weights_sum) * self.image_group1.phase_spectrum
                
            if  self.image_group2.combo_box.currentText() == "Magnitude":
                self.mixed_magnitude+= (weight_2 / weights_sum) * self.image_group2.magnitude_spectrum
            else:
                self.mixed_phase+= (weight_2 / weights_sum) * self.image_group2.phase_spectrum
            
            if  self.image_group3.combo_box.currentText() == "Magnitude":
                self.mixed_magnitude+= (weight_3 / weights_sum) * self.image_group3.magnitude_spectrum
            else:
                self.mixed_phase+= (weight_3 / weights_sum) * self.image_group3.phase_spectrum
            
            if  self.image_group4.combo_box.currentText() == "Magnitude":
                self.mixed_magnitude+= (weight_4 / weights_sum) * self.image_group4.magnitude_spectrum
            else:
                self.mixed_phase+= (weight_4 / weights_sum) * self.image_group4.phase_spectrum
        else:
            if  self.image_group1.combo_box.currentText() == "Real":
                self.mixed_real+= (weight_1 / weights_sum) * self.image_group1.real_spectrum
            else:
                self.mixed_imaginary+= (weight_1 / weights_sum) * self.image_group1.imaginary_spectrum
                
            if  self.image_group2.combo_box.currentText() == "Real":
                self.mixed_real+= (weight_2 / weights_sum) * self.image_group2.real_spectrum
            else:
                self.mixed_imaginary+= (weight_2 / weights_sum) * self.image_group2.imaginary_spectrum
            
            if  self.image_group3.combo_box.currentText() == "Real":
                self.mixed_real+= (weight_3 / weights_sum) * self.image_group3.real_spectrum
            else:
                self.mixed_imaginary+= (weight_3 / weights_sum) * self.image_group3.imaginary_spectrum
            
            if  self.image_group4.combo_box.currentText() == "Real":
                self.mixed_real+= (weight_4 / weights_sum) * self.image_group4.real_spectrum
            else:
                self.mixed_imaginary+= (weight_4 / weights_sum) * self.image_group4.imaginary_spectrum
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWidget()
    ex.show()
    sys.exit(app.exec_())