#Usage
#Navigate with arrow keys. Press "Space" once to start a new sequence and press it again to end the sequence.
#The sequence will be all images between the two images where space was clicked including the images themselves 
#(you can go back and forth).
# On the left, you can see the last picture to compare. The button on the bottom saves everything to txt.

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class ImageLabeler(QWidget):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.index = 1
        self.sequences = []
        self.current_sequence = []
        self.marking_sequence = False
        self.start = None
        self.end = None
        
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.controls_layout = QHBoxLayout()
        self.label_current = QLabel(self)
        self.label_previous = QLabel(self)
        self.label_info = QLabel(f"Remaining Images: {len(self.images) - self.index}", self)

        self.save_button = QPushButton('Save Sequences', self)
        self.save_button.clicked.connect(self.saveSequences)

        self.image_layout.addWidget(self.label_previous)
        self.image_layout.addWidget(self.label_current)
        self.layout.addLayout(self.image_layout)
        self.controls_layout.addWidget(self.label_info)
        self.controls_layout.addWidget(self.save_button)
        self.layout.addLayout(self.controls_layout)

        self.setLayout(self.layout)

        self.updateImages()
        self.setWindowTitle('Image Labeler')
        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right and self.index < len(self.images) - 1:
            self.index += 1
            self.updateImages()
            self.addToSequence()
        elif event.key() == Qt.Key_Left and self.index > 1:
            self.index -= 1
            self.updateImages()
            self.addToSequence()

        elif event.key() == Qt.Key_Space:
            self.handleSpacePress()

    def updateImages(self):
        current_image_path = os.path.join(self.folder_path, self.images[self.index])
        previous_image_path = os.path.join(self.folder_path, self.images[self.index - 1])

        current_image = QImage(current_image_path)
        previous_image = QImage(previous_image_path)
        screen_size = self.screen().size()
        scaled_current_image = current_image.scaled(screen_size.width() // 2, screen_size.height(), Qt.KeepAspectRatio)
        scaled_previous_image = previous_image.scaled(screen_size.width() // 2, screen_size.height(), Qt.KeepAspectRatio)

        self.label_current.setPixmap(QPixmap.fromImage(scaled_current_image))
        self.label_previous.setPixmap(QPixmap.fromImage(scaled_previous_image))
        self.label_info.setText(f"Remaining Images: {len(self.images) - self.index - 1}")

    def handleSpacePress(self):
        if not self.marking_sequence:
            self.current_sequence = [self.images[self.index]]
            self.start = self.images[self.index]
            self.marking_sequence = True
        else:
            self.current_sequence = []
            self.marking_sequence = False
            self.end = self.images[self.index]
            current_sequence = self.elements_between(self.images, self.start, self.end)
            self.sequences.append(current_sequence)
            print("Sequence stored:", self.sequences)
    def elements_between(self,arr, start_str, end_str):
        try:
            start_index = arr.index(start_str)
            end_index = arr.index(end_str)
            if start_index > end_index:
                start_index, end_index = end_index, start_index
            return arr[start_index:end_index+1]
        except ValueError:
            return []

    def addToSequence(self):
        if self.marking_sequence:
            if self.images[self.index] not in self.current_sequence:
                self.current_sequence.append(self.images[self.index])
    def saveSequences(self):
        with open('image_sequences.txt', 'w') as file:
            for sequence in self.sequences:
                file.write(', '.join(sequence) + '\n')
        print("All sequences have been saved to image_sequences.txt")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    path_to_images = '/Users/lukaslaskowski/Documents/HPI/gorillavision/data/kwitonda/all_images'
    ex = ImageLabeler(path_to_images)
    sys.exit(app.exec_())