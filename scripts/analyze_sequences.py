import sys
import random
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ImageWidget(QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.labels = []

        # Get image paths
        image_paths = [os.path.join(self.folder_path, f) for f in os.listdir(self.folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_images = random.sample(image_paths, min(3, len(image_paths)))

        for path in selected_images:
            label = QLabel(self)
            pixmap = QPixmap(path)
            label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))
            self.labels.append(label)
            layout.addWidget(label)

        folder_name = QLabel(os.path.basename(self.folder_path))
        layout.addWidget(folder_name, alignment=Qt.AlignCenter)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self, folder_of_folders):
        super().__init__()
        self.folder_of_folders = folder_of_folders
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Cluster Viewer")
        self.grid_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.grid_layout)
        self.setCentralWidget(self.central_widget)

        self.folder_paths = [os.path.join(self.folder_of_folders, f) for f in sorted(os.listdir(self.folder_of_folders), key=lambda x: int(x)) if os.path.isdir(os.path.join(self.folder_of_folders, f))]
        self.widgets = []
        self.current_index = 0

        self.load_images(self.current_index)

    def load_images(self, start):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for i in range(6):
            idx = (start + i) % len(self.folder_paths)
            widget = ImageWidget(self.folder_paths[idx])
            self.grid_layout.addWidget(widget)
            self.widgets.append(widget)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Right:
            self.current_index = (self.current_index + 1) % len(self.folder_paths)
            self.load_images(self.current_index)
        elif e.key() == Qt.Key_Left:
            self.current_index = (self.current_index - 1 + len(self.folder_paths)) % len(self.folder_paths)
            self.load_images(self.current_index)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    path_to_folders = '/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/sequences_cropped'
    ex = MainWindow(path_to_folders)
    ex.show()
    sys.exit(app.exec_())
