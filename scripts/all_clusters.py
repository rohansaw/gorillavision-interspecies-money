import sys
import os
import random
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class PaginatedClusterViewer(QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.current_page = 0
        self.images_per_page = 4  # Set number of images per page
        self.initUI()

    def initUI(self):
        self.screen = QDesktopWidget().screenGeometry()
        self.setGeometry(0, 0, self.screen.width(), self.screen.height())
        
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
        self.setWindowTitle('Cluster Image Fullscreen Viewer')
        self.showFullScreen()
        self.loadPage()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            if self.current_page < self.getTotalPages() - 1:
                self.current_page += 1
                self.loadPage()
        elif event.key() == Qt.Key_Left:
            if self.current_page > 0:
                self.current_page -= 1
                self.loadPage()

    def loadPage(self):
        # Clear current grid layout
        for i in reversed(range(self.grid_layout.count())): 
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        start_index = self.current_page * self.images_per_page
        end_index = start_index + self.images_per_page
        folders_to_display = self.folders[start_index:end_index]

        for i, folder in enumerate(folders_to_display):
            current_folder = os.path.join(self.folder_path, folder)
            images = [img for img in os.listdir(current_folder) if os.path.isfile(os.path.join(current_folder, img))]
            if images:
                selected_image = random.choice(images)
                pixmap = QPixmap(os.path.join(current_folder, selected_image))

                label = QLabel()
                label.setPixmap(pixmap.scaled(self.screen.width() // 2, self.screen.height() // 2, Qt.KeepAspectRatio))
                label.setAlignment(Qt.AlignCenter)
                caption = QLabel(folder)
                caption.setAlignment(Qt.AlignCenter)

                self.grid_layout.addWidget(label, i // 2, i % 2)
                self.grid_layout.addWidget(caption, i // 2 + 1, i % 2)

    def getTotalPages(self):
        return (len(self.folders) + self.images_per_page - 1) // self.images_per_page

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PaginatedClusterViewer('/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/sequences_cropped')
    sys.exit(app.exec_())
