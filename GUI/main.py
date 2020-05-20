import sys
sys.path.append("..")

import os
from os.path import join
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QIcon, QPixmap, QImage  
import numpy as np
import matplotlib.pyplot as plt
import cv2
from evaluation import evaluation
import numbergen as ng
import imagen as ig

class App(QWidget):

    def __init__(self):
        super(App, self).__init__()
        self.title = 'Soft Shadow Generation'
        self.left = 100
        self.top = 100
        self.width = 1280
        self.height = 480
        
        self.template_label = QLabel(self)

        self.cutout_label = QLabel(self)
        self.cutout_label.move(256, 0)

        self.ibl_label = QLabel(self)
        self.ibl_label.move(256 * 2, 0)
        
        self.next_btn = QPushButton("next")
        self.next_btn.move(256 * 4, 0)
        self.next_btn.clicked.connect(self.next_scene)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.template_label)
        self.layout.addWidget(self.cutout_label)
        self.layout.addWidget(self.ibl_label)
        self.layout.addWidget(self.next_btn)

        self.setLayout(self.layout)

        self.cur_ibl = np.zeros((256,512,3))

        self.ibl_cmds = []
        self.init_ibl_state()
        self.add_cur_ibl_state()

        # test imgs
        self.num_test = 3 
        self.final_imgs = [join('imgs', '{:03d}_final.png'.format(i)) for i in range(1,self.num_test+1)]
        self.mask_imgs = [join('imgs','{:03d}_mask.png'.format(i)) for i in range(1,self.num_test+1)]
        self.cutout_imgs = [join('imgs','{:03d}_cutout.png'.format(i)) for i in range(1,self.num_test+1)]
        self.cur_exp = 0

        self.initUI()
    
    def init_ibl_state(self):
        self.cur_x, self.cur_y, self.cur_size = 0.0, 0.0, 0.1

    def reset_ibl_cmds(self):
        self.ibl_cmds = []
        self.init_ibl_state()

    def add_cur_ibl_state(self):
        self.ibl_cmds.append((self.cur_x, self.cur_y, self.cur_size))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.update_cur_labels()
        self.show()

    def set_image(self, label, img_path):
        # Create widget
        pixmap = QPixmap(img_path)
        label.setPixmap(pixmap)

    def set_np_img(self, label, img_np):
        if len(img_np.shape) == 2:
            img_np = np.repeat(img_np[:,:,np.newaxis], 3, axis=2)

        pixmap = QPixmap(self.to_qt_img(img_np))
        label.setPixmap(pixmap)

    def to_qt_img(self, np_img):
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img, 0.0, 1.0)
            np_img = np_img * 255.0
            np_img = np_img.astype(np.uint8)

        h,w,c = np_img.shape
        # bytesPerLine = 3 * w
        return QImage(np_img.data, w,h, QImage.Format_RGB888)

    def update_cur_labels(self):
        self.cur_final, self.cur_cutout, self.cur_mask = self.load_cur_np(self.cur_exp)
        self.set_np_img(self.template_label, self.cur_final)
        self.set_np_img(self.cutout_label, self.cur_cutout)
        self.set_np_img(self.ibl_label, self.cur_ibl)
        self.ibl_label.setFocus()

    def next_scene(self):
        print('go to next scene')
        self.cur_exp += 1
        self.cur_exp = self.cur_exp % self.num_test
        print(self.cur_exp)
        self.update_cur_labels()

    def load_cur_np(self, cur_exp):
        final, cutout, mask = cv2.imread(self.final_imgs[cur_exp]), cv2.imread(self.cutout_imgs[cur_exp]), cv2.imread(self.mask_imgs[cur_exp])
        return final, cutout, mask

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        if event.key() == QtCore.Qt.Key_Q:
            print('increase size')
            self.cur_size += 0.005
            self.cur_size = np.clip(self.cur_size, 0.001, 1.0)
            self.update_composite()

        if event.key() == QtCore.Qt.Key_W:
            print('decrease size')
            self.cur_size -= 0.005
            self.cur_size = np.clip(self.cur_size, 0.001, 1.0)
            self.update_composite()

        if event.key() == QtCore.Qt.Key_A:
            print('add an area light')
            self.add_cur_ibl_state()
            self.init_ibl_state()
            self.update_composite()

    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.NoButton:
            print("Simple mouse motion")
        elif event.buttons() == QtCore.Qt.LeftButton:
            print("Left click drag")

        elif event.buttons() == QtCore.Qt.RightButton:
            print("Right click drag")
        super(App, self).mouseMoveEvent(event)

    def set_ibl(self, ibl_np):
        self.set_np_img(self.ibl_label, ibl_np)
    
    def composite(self, final, mask, shadow):
        shadow = np.clip(shadow, 0.0, 1.0)
        composite = mask * final + (1.0 - mask) * (1.0-shadow)
        return composite

    def update_composite(self):
        self.ibl_cmds[-1] = (self.cur_x, self.cur_y, self.cur_size)

        cur_ibl = self.get_cur_ibl()
        self.set_ibl(cur_ibl)

        shadow_result = evaluation.net_render_np(self.cur_mask, cur_ibl)
        shadow_result = np.repeat(shadow_result[:,:,np.newaxis], 3, axis=2)
        final_composite = self.composite(self.cur_final/255.0, self.cur_mask/255.0, shadow_result)
        self.set_np_img(self.cutout_label, final_composite)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            print("Press! {}".format(event.pos()))
            x,y = self.get_relative_pos(event)
            self.cur_x, self.cur_y = x, y
            self.update_composite()

        super(App, self).mousePressEvent(event)

    def get_relative_pos(self, mouse_event):
        ibl_pos = self.ibl_label.pos()
        mouse_pos = mouse_event.pos()
        ibl_w, ibl_h = 512, 256
        relative = mouse_pos - ibl_pos
        x,y = relative.x()/ibl_w, 1.0 - (relative.y() - 100)/ibl_h
        print('x: {}, y: {}'.format(x,y))
        return x,y 

    def get_cur_ibl(self):
        num = len(self.ibl_cmds)

        gs = ig.Composite(operator=np.add,
                generators=[ig.Gaussian(
                            size=self.ibl_cmds[i][2],
                            scale=1.0,
                            x=self.ibl_cmds[i][0]-0.5,
                            y=self.ibl_cmds[i][1]-0.5,
                            aspect_ratio=1.0,
                            ) for i in range(num)],
                    xdensity=512)
        return np.repeat(gs()[:,:,np.newaxis], 3, axis=2) 

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
