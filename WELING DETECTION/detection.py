import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QTableWidget, \
    QTableWidgetItem, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
from ultralytics import YOLO

#инициализация YOLO
weights_path = r'C:\hakaton\WELING DETECTION\weight\best.pt'
model = YOLO(weights_path)

#обнаружение обьектов на изображении
def detect(image_path):
    global goodwork, notdetect
    img = cv2.imread(image_path)

    results = model(img)
    if not results or all(len(result.boxes) == 0 for result in results):
        notdetect = True
        print("not detected")
        return img, []
    else:
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cls_idx = int(box.cls[0])
                class_name = model.names[cls_idx]
                if class_name != "Good Welding":
                    goodwork = False
                detections.append((class_name, (x1, y1, x2, y2)))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, class_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    print(goodwork)
    return img, detections

#интерфейс
class DetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.loadButton = QPushButton('Load Image')
        self.loadButton.clicked.connect(self.load_image)
        self.layout.addWidget(self.loadButton)

        self.detectButton = QPushButton('Run Detection')
        self.detectButton.clicked.connect(self.run_detection)
        self.layout.addWidget(self.detectButton)

        self.imageLabel = QLabel()
        self.imageLabel.setScaledContents(False)
        self.layout.addWidget(self.imageLabel)

        self.resultLabel = QLabel()
        self.layout.addWidget(self.resultLabel)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        self.setLayout(self.layout)
        self.setWindowTitle('Image Detection App')

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)",
                                                  options=options)
        if fileName:
            self.imagePath = fileName
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(False)
            self.imageLabel.setAlignment(Qt.AlignCenter)

    def run_detection(self):
        if hasattr(self, 'imagePath'):
            global notdetect, goodwork
            notdetect = False
            goodwork = True
            img, detections = detect(self.imagePath)
            if notdetect:
                self.show_no_detection_message()
                self.table.clear()
                self.table.setRowCount(0)
                self.table.setColumnCount(0)
                self.resultLabel.setText("No Detection")
            else:
                output_path = r'C:\hakaton\WELING DETECTION\images\output\output.jpg'
                cv2.imwrite(output_path, img)
                pixmap = QPixmap(output_path)
                self.imageLabel.setPixmap(pixmap)
                self.imageLabel.setScaledContents(False)
                self.imageLabel.setAlignment(Qt.AlignCenter)

                if goodwork:
                    self.resultLabel.setText("Detection: Good Welding")
                else:
                    self.resultLabel.setText("Detection: Bad Welding")

                self.table.setRowCount(len(detections))
                self.table.setColumnCount(2)
                self.table.setHorizontalHeaderLabels(['Class', 'Coordinates'])
                for i, detection in enumerate(detections):
                    class_name, coords = detection
                    self.table.setItem(i, 0, QTableWidgetItem(class_name))
                    coordinates_item = QTableWidgetItem(f'({coords[0]}, {coords[1]}), ({coords[2]}, {coords[3]})')
                    coordinates_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    self.table.setItem(i, 1, coordinates_item)
                self.table.resizeColumnsToContents()
        else:
            self.resultLabel.setText("Please load an image first")

    def show_no_detection_message(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Not detected")
        msg.setWindowTitle("Detection Result")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def resizeEvent(self, event):
        self.update_image()

    def update_image(self):
        if hasattr(self, 'imagePath'):
            pixmap = QPixmap(self.imagePath)
            self.imageLabel.setPixmap(
                pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DetectionApp()
    ex.show()
    sys.exit(app.exec_())
