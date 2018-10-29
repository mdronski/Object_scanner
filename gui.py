import io
import sys

from PyQt5.QtCore import QThread, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QImage
import numpy as np
from matplotlib import pyplot as plt





class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 image - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 638
        self.height = 478
        self.image_stream = open("python_gui", mode='rb')
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.init_display_window()
        self.initialise_timer()
        # self.image_updater = self.ImageUpdater(label=self.label, parent=self)
        # self.image_updater.run()


    def init_display_window(self):
        self.label = QLabel(self)
        pixmap = QPixmap('filtered.ppm')
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignTop)
        self.resize(self.width, self.height)
        self.show()

    def refresh(self):
        data = self.image_stream.read()
        qim = QImage(data, self.width, self.height, QImage.Format_RGB888)
        # print(qim.height(), qim.width())
        self.label.setPixmap(QPixmap.fromImage(qim))
        self.label.repaint()


    def initialise_timer(self):
        timer = QTimer(self)
        timer.timeout.connect(self.refresh)
        timer.start(1)




    # def refresh_image(self):
    #
    #     FIFO = 'python_gui'
    #     with open(FIFO, mode='rb') as fifo:
    #         print("FIFO opened")
    #         data = fifo.read()
    #         if len(data) == 0:
    #             print("Writer closed")
    #         print(type(data))
    #         A = np.asarray(list(data))
    #         print(A)
    #         qim = QImage(data, 638, 478, QImage.Format_RGB888)
    #         print(qim.height(), qim.width())
    #         self.label.setPixmap(QPixmap.fromImage(qim))
    #         self.repaint()



    # class ImageUpdater(QThread):
    #
    #     def __init__(self, label, parent):
    #         # super.__init__(self)
    #         super(self.__class__, self).__init__(parent)
    #         self.label = label
    #         self.parent1 = parent
    #
    #     def run(self):
    #         FIFO = 'python_gui'
    #         with open(FIFO, mode='rb') as fifo:
    #             print("FIFO opened")
    #             while(True):
    #                 data = fifo.read()
    #                 if len(data) == 0:
    #                     print("Writer closed")
    #                     break
    #                 qim = QImage(data, 638, 638, QImage.Format_RGB888)
    #                 print(qim.height(), qim.width())
    #                 self.label.setPixmap(QPixmap.fromImage(qim))
    #                 self.label.repaint()
    #






if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

# FIFO = 'python_gui'
# with open(FIFO, mode='rb') as fifo:
#     print("FIFO opened")
#     while True:
#         print('xd')
#         data = fifo.read()
#         print('xd')
#         if len(data) == 0:
#             print("Writer closed")
#             break
#         A = b'P6\n640 640\n255\n' + data
#         print('xd')
#         file = open('python-image.ppm', 'wb')
#         print('xd')
#         file.write(A)
#         print('xd')
#         file.close()
#         break
#

