# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
from PyQt5.QtWidgets import QApplication ,QMainWindow ,QPushButton ,QWidget ,QListWidget ,QLabel ,QListView ,QMessageBox
from Image_utility_tool import Ui_MainWindow
from Image_plot import Ui_plot_image
from PyQt5 import QtWidgets, QtCore
import sys
import shutil
import os
import hashlib
import progressbar
from PyQt5.QtGui import QPixmap, QPalette
from tkinter import messagebox
from PIL import Image , ImageDraw
from tkinter import filedialog
from tkinter import *
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene, QGraphicsView

class page(Ui_plot_image, QMainWindow):
    def __init__(self,table):
        super().__init__()
        self.setupUi(self)
        self.table = table
        self.Button_next.clicked.connect(self.next_image)
        self.Button_prev.clicked.connect(self.prev_image)
        self.Button_del.clicked.connect(self.delete_image)
        self.Next_row_im.clicked.connect(self.Next_row_image)
        self.Prev_row_im.clicked.connect(self.Prev_row_image)
        self.r = None
        self.c = None



    def next_image(self):
        r_c = self.table.columnCount()
        if self.r == None and self.c == None:
            self.c = self.table.currentColumn()
            self.r = self.table.currentRow()
        self.c = self.c +1
        if self.c + 1 == self.column_max :
           self.Button_prev.show()
           self.Button_next.hide()
        elif self.c == 0:
            self.Button_prev.hide()
            self.Button_next.show()
        else:
            self.Button_prev.show()
            self.Button_next.show()
        try:
            self.update_image(os.path.normpath(self.duplicates2[self.r][self.c]))

        except:
            messagebox.showinfo(title='Error massage', message='empty cell')


    def prev_image(self):
        r_c = self.table.columnCount()
        if self.r == None and self.c == None:
            self.c = self.table.currentColumn()
            self.r = self.table.currentRow()
        self.c = self.c -1
        if self.c + 1 == self.column_max :
           self.Button_prev.show()
           self.Button_next.hide()
        elif self.c == 0:
            self.Button_prev.hide()
            self.Button_next.show()
        else:
            self.Button_prev.show()
            self.Button_next.show()

        try:
            self.update_image(os.path.normpath(self.duplicates2[self.r][self.c]))
        except:
            messagebox.showinfo(title='Error massage', message='empty cell')

    def delete_image(self):
        try:
            os.remove(self.f_path)
        except:
            messagebox.showinfo(title='Error massage', message='Image is not exist')

    def update_image(self,path):
        geo = self.label.geometry().getRect()
        pixmap = QPixmap(path)
        pixmap_resized = pixmap.scaled(geo[3], geo[3])
        self.label_2.setText("Image path: " + path)
        self.label.setPixmap(pixmap_resized)
        self.label.setScaledContents(True)
        tmp = path.split("\\")
        self.label_3.setText("Class lable: " + tmp[-2])
        self.label_3.setStyleSheet("color : red")

    def Next_row_image(self):
        r_c = self.table.rowCount()
        if self.r == None and self.c == None:
            self.c = self.table.currentColumn()
            self.r = self.table.currentRow()

        if (self.r + 1)!= r_c:
            self.r = self.r +1
            try:
                self.f_path = os.path.normpath(self.duplicates2[self.r][self.c])

                geo = self.label.geometry().getRect()
                pixmap = QPixmap(self.f_path)
                pixmap_resized = pixmap.scaled(geo[3], geo[3])
                self.label_2.setText("Image path: " + self.f_path)
                self.label.setPixmap(pixmap_resized)
                self.label.setScaledContents(True)
            except:
                messagebox.showinfo(title='Error massage', message='empty cell')
            tmp = self.f_path.split("\\")
            self.label_3.setText("Class lable: " + tmp[-2])
            self.label_3.setStyleSheet("color : red")
        else:
            messagebox.showinfo(title='Error massage', message='end of the image list')

    def Prev_row_image(self):
        r_c = self.table.rowCount()
        if self.r == None and self.c == None:
            self.c = self.table.currentColumn()
            self.r = self.table.currentRow()

        if self.r != 0:
            self.r = self.r - 1
            try:
                self.f_path = os.path.normpath(self.duplicates2[self.r][self.c])
                geo = self.label.geometry().getRect()
                pixmap = QPixmap(self.f_path)
                pixmap_resized = pixmap.scaled(geo[3], geo[3])
                self.label_2.setText("Image path: " + self.f_path)
                self.label.setPixmap(pixmap_resized)
                self.label.setScaledContents(True)
            except:
                messagebox.showinfo(title='Error massage', message='empty cell')
            tmp = self.f_path.split("\\")
            self.label_3.setText("Class lable: " + tmp[-2])
            self.label_3.setStyleSheet("color : red")
        else:
            messagebox.showinfo(title='Error massage', message='head of the image list')

class ValueItem(QGraphicsItem):
    def __init__(self, value):
        super().__init__()
        self._value = value
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def boundingRect(self):
        return QRectF(-20, -20, 40, 40)

    def paint(self, painter, option, widget=None):
        painter.setBrush(QColor(255, 0, 0))
        painter.drawRect(-20, -20, 40, 40)
        painter.drawText(QRectF(-20, -20, 40, 40), str(self._value))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            print(f"Position changed to {value}")
        return super().itemChange(change, value)

    def setValue(self, value):
        self._value = value
        self.update()

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Converted Image'
        self.left = 800
        self.top = 500
        self.width = 600
        self.height = 600
        #self.messagebox()

    def messagebox(self, im):
        img = Image.open(im)
        w,h = img.size
        d = ImageDraw.Draw(img)
        d.ink = 1
        d.rectangle([(w/2+100,h/2+100),(w/2-100,h/2-100)],outline="black",width = 10)
        temp_image_path = os.path.normpath(os.getcwd() + "/temp.jpeg")
        img.save(temp_image_path)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        label = QLabel(self)
        pixmap = QPixmap(temp_image_path)
        pixmap_resized = pixmap.scaled(600, 600)
        label.setPixmap(pixmap_resized)
        #self.resize(pixmap.width(), pixmap.height())
        self.show()
        Reply = QMessageBox.question(self, 'Converted Image', "Does the offset is correct?", QMessageBox.Yes | QMessageBox.No,QMessageBox.No)
        if Reply == QMessageBox.Yes:
            print('Yes clicked.')
            x = True
        else:
            print('No clicked.')
            x = False
        self.close()
        return x

class UI(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.init_button_actions()

        self.sub_window = page(self.tableWidget)
        self.Image_view = App()

    # initialize all functions
    def init_button_actions(self):
        self.setWindowTitle("ADC utility tool")
        self.offset_y.hide()
        self.offset_x.hide()
        self.x_offset_text.hide()
        self.y_offset_text.hide()
        self.hist_min.hide()
        self.hist_max.hide()
        self.hist_value.hide()
        self.hist_value_2.hide()
        self.hist_upper.hide()
        self.hist_lower.hide()
        self.horizontalSlider_2.hide()
        self.horizontalSlider.hide()
        self.graphicsView.hide()
        self.radioButton.clicked.connect(self.offset_set)
        self.source_button.clicked.connect(self.select_s)
        self.destination_button.clicked.connect(self.select_d)
        self.commandLinkButton.clicked.connect(self.image_convert_and_crop)
        self.Star_seperate.clicked.connect(self.seperate_images)
        self.seek_identical.clicked.connect(self.find_duplicates)
        self.tableWidget.clicked.connect(self.table_clicked)
        self.Extractor_next.clicked.connect(self.next_file_in_list)
        self.Extractor_prev.clicked.connect(self.prev_file_in_list)
        self.checkBox_2.clicked.connect(self.histogram_view)
        self.horizontalSlider.valueChanged.connect(self.lower_th_display)
        self.horizontalSlider_2.valueChanged.connect(self.upper_th_display)
        self.horizontalScrollBar.valueChanged.connect(self.scrollbar)
        self.scale_list=np.linspace(0.1,1,num=50)

    def histogram_view(self):
        if self.checkBox_2.isChecked():
            self.horizontalSlider.show()
            self.horizontalSlider_2.show()
            self.graphicsView.show()
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 530, 591))
            self.seek_identical.setText('Analyz images backgroung')
            self.hist_min.show()
            self.hist_max.show()
            self.hist_value.show()
            self.hist_value_2.show()
            self.hist_upper.show()
            self.hist_lower.show()
        else:
            self.horizontalSlider.hide()
            self.graphicsView.hide()
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 881, 591))
            self.seek_identical.setText('Find identical images')
            self.hist_min.hide()
            self.hist_max.hide()
            self.hist_value.hide()
            self.hist_value_2.hide()
            self.hist_upper.hide()
            self.hist_lower.hide()
            self.horizontalSlider_2.hide()

    #zoom in scroll
    def wheelEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            numPixels = event.pixelDelta()
            scrollDistance = event.angleDelta().y()
            numDegrees = event.angleDelta() / 8
            numSteps = numDegrees / 15
            if scrollDistance > 0:
                zoom_indx = -1*numSteps.manhattanLength()
            else:
                zoom_indx = numSteps.manhattanLength()
            s_indx = self.horizontalScrollBar.value()
            #self.horizontalScrollBar.
            self.horizontalScrollBar.setValue(s_indx + zoom_indx)

    # mouse clicked on image
    def mousePressEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            if self.destination_TextEdit.toPlainText() == "":
                messagebox.showinfo(title='Error massage', message='please select destination folder')
            else:
                rect = self.label_5.geometry().getRect()
                x = event.x() - rect[0] -5
                y = event.y() - rect[1] - 45
                img_dim = Image.open(self.current_image_path).size
                x_new = (img_dim[0]/rect[2])*x
                y_new = (img_dim[1]/rect[3])*y
                offset_y  = int((img_dim[1] / 2 ) - y_new)
                offset_x = int((img_dim[0] / 2) - x_new)
                top = math.ceil((img_dim[0]))
                left = math.ceil((img_dim[1]))
                img = Image.open(self.current_image_path)
                img1 = Image.new(img.mode, (img_dim[0] + abs(offset_x), img_dim[1] + abs(offset_y)))
                img1.paste(img, (0 + offset_x, 0 + offset_y))
                img2 = img1.crop((0,0,img_dim[0],img_dim[1]))
                tmp_str = self.current_image_path.split('\\')
                img_name=tmp_str[-1]
                while os.path.exists(self.destination_path + '\\_' + img_name) or os.path.exists(self.destination_path + '\\' + img_name):
                    tmp = img_name.split(".jpeg")
                    img_name = tmp[0] + '_' + '.jpeg'
                img2.save(self.destination_path + '\\_' + img_name)
                self.Log_listwidget.addItem(img_name + ' image was saved')
                self.Log_listwidget.scrollToBottom()
                self.next_file_in_list()
                # complete saving and next image ------------------- to dooooo

    #show offset fields
    def offset_set(self):
        if self.radioButton.isChecked():
            self.offset_y.show()
            self.offset_x.show()
            self.x_offset_text.show()
            self.y_offset_text.show()
        else:
            self.offset_y.hide()
            self.offset_x.hide()
            self.x_offset_text.hide()
            self.y_offset_text.hide()

    #show next image in list
    def next_file_in_list(self):

        list = self.f_object[0]
        file_indx = self.f_object[1] + 1
        try:
            self.current_image_path = list[file_indx]
            self.f_object[1] = file_indx
            l = len(list)
            self.Log_listwidget.addItem("next image " + str(file_indx) + "/" + str(l))
            self.Log_listwidget.scrollToBottom()
            if l - 1 == file_indx :
                self.Extractor_next.hide()
            self.image_changing(self.current_image_path)
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            geo = self.label_5.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_5.setPixmap(pixmap)
        except:
            QMessageBox.about(self, "info massage", "no more images")

    def image_changing(self,path):
        zoom = self.horizontalScrollBar.value()
        i = Image.open(path)
        im_size = i.size
        if im_size[0] != im_size[1]:
            i = self.padding_image(path)
            im_size = i.size
        if zoom != 50:
            if zoom < 50:
                scale = self.scale_list[zoom]
            elif zoom == 0:
                scale = 0.1
            else:
                scales = self.scale_list[zoom - 50] * 10
            left = im_size[0] * (1 - scale) / 2
            right = im_size[0] - left
            top = im_size[1] * (1 - scale) / 2
            bot = im_size[1] - top
            i = i.crop((left, top, right, bot))
            i = i.resize(im_size, Image.ANTIALIAS)
        draw = ImageDraw.Draw(i)
        draw.rectangle([(im_size[0] / 2 - 100, im_size[1] / 2 + 100), (im_size[0] / 2 + 100, im_size[1] / 2 - 100)],
                       outline="black", width=8)
        i.save(os.getcwd() + '\\tmp.jpeg')

    def padding_image(self,path):
        # Define the padding values.
        image =  Image.open(path)
        im_size = image.size
        padding = int(abs(im_size[0] - im_size[1]) / 2)
        padding_values = (padding, padding, 0, 0)
        top, bottom, left, right = padding_values
        # Get the image shape: (width, height)
        width, height = image.size
        # Compute the size of the padded image. How each dimension will change.
        width2, height2 = width + left + right, height + top + bottom
        # Get the full image area after padding. This will be an image filled with the
        # padding color.
        padding_color = (0, 0, 0)
        padded_image = Image.new(mode="RGB", size=(width2, height2), color=padding_color)
        # paste the original image to the padding area defined above.
        # box - tuple giving the top left corner.
        padded_image.paste(image, box=(left, top))
        return padded_image

    # show next image in list
    def prev_file_in_list(self):
        self.Extractor_next.show()
        list = self.f_object[0]
        file_indx = self.f_object[1] -1
        self.current_image_path = list[file_indx]
        self.f_object[1] = file_indx
        l = len(list)
        self.Log_listwidget.addItem("next image " + str(file_indx) + "/" + str(l))
        self.Log_listwidget.scrollToBottom()
        if file_indx == 0:
            self.Extractor_next.hide()
        self.image_changing(self.current_image_path)
        pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
        geo = self.label_5.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_5.setPixmap(pixmap)

    #seperate images into subfolders
    def seperate_images(self):
        widgets = ['Loading: ', progressbar.AnimatedMarker()]
        bar = progressbar.ProgressBar(widgets=widgets).start()
        self.progressBar.setValue(0)
        path = self.Source_path
        if not self.checkBox.isChecked():
            try:
                N = int(self.plainTextEdit.toPlainText())  # number of images per subfolder
                self.Log_listwidget.addItem("start seperation into subfolders - each folder contains: " + str(N))
                self.Log_listwidget.scrollToBottom()

                self.plainTextEdit_2.setPlainText('in progress')
                counter = 0
                folder_name = "folder"
                folder_path = os.path.join(self.destination_path, folder_name + str(counter))
                os.makedirs(folder_path)

                for root, dirs, files in os.walk(path):
                    l  = len(files)
                    for i ,file in enumerate(files):
                        self.progressBar.setValue(i/l)
                        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                            if len(os.listdir(folder_path)) >= N:
                                counter += 1
                                folder_name = "folder"
                                folder_path = os.path.join(self.destination_path, folder_name + str(counter))
                                os.makedirs(folder_path)
                                self.Log_listwidget.addItem("subfolder: " + str(counter) + "ready")
                            shutil.copy(os.path.join(root, file), os.path.join(folder_path, file))
                self.plainTextEdit_2.setPlainText('finish')
                self.Log_listwidget.addItem("finished seperat into: " + str(counter) + "subfolders" )
                self.Log_listwidget.scrollToBottom()
            except:
                QMessageBox.about(self, "Error msg", "please selects input folder")
        else:
            #path = 'A:/bes data/X5 recipe/a/BES-Reports-for-classification/BA059PD-TSV-N-BES/Setup1/4244209'
            #self.destination_path ='A:\\bes data\\X5 recipe\\pre_db'
            path = self.Source_TextEdit.toPlainText()
            l = len(path)+1

            for root, dirs, files in os.walk(path):
                folder_hir_list = []
                copy_path = self.destination_path
                if 'ADC' in dirs:

                    tmp_str1 = root[l:]
                    tmp_str = tmp_str1.split('\\')
                    for string in tmp_str:
                        if not os.path.exists(copy_path + '\\' + string):
                            folder_hir_list.append(string)
                            #os.mkdir(copy_path + '\\' + string)
                        copy_path = os.path.normpath(copy_path + '\\' + string)
                    copy_path =os.path.normpath(copy_path + '\\ADC')
                    #os.mkdir(copy_path)
                    file_list = os.listdir(root + '\\ADC')
                    dataframe1 = pd.read_csv(root + '\\ADC\\' + 'Surface2Bump.csv')
                    #dataframe1.sort_values('Recipe')
                    img_path_list =  dataframe1['defect_key'].values
                    recipe_list = dataframe1['Recipe'].values
                    indx = 0
                    p = self.destination_path
                    for i,file in enumerate(img_path_list):
                        try:
                            curr_path = os.path.normpath(self.destination_path + '\\' + str(recipe_list[i]) + '\\' + tmp_str1)
                            if not os.path.exists(curr_path):
                                if not os.path.exists(curr_path):
                                    if not os.path.exists(self.destination_path + '\\' + str(recipe_list[i])):
                                        os.mkdir(self.destination_path + '\\' + str(recipe_list[i]))
                                tmp_path = self.destination_path + '\\' + str(recipe_list[i])
                                for folder in tmp_str:
                                    if not os.path.exists(tmp_path + '\\' +  folder):
                                        os.mkdir(tmp_path + '\\' +  folder)
                                    tmp_path = tmp_path + '\\' +  folder
                                os.mkdir(tmp_path + '\\ADC\\')

                                shutil.copyfile(os.path.normpath(root + '\\ADC\\image_flow.csv'), os.path.normpath(
                                    self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\image_flow.csv'))

                                shutil.copyfile(os.path.normpath(root + '\\ADC\\image_view.csv'), os.path.normpath(
                                    self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\image_view.csv'))

                                shutil.copyfile(os.path.normpath(root + '\\ADC\\ManReClassify.ini'), os.path.normpath(
                                    self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\ManReClassify.ini'))

                                shutil.copyfile(os.path.normpath(root + '\\ADC\\run_details.json'), os.path.normpath(
                                    self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\run_details.json'))

                                shutil.copyfile(os.path.normpath(root + '\\ADC\\Surface2Bump.csv'), os.path.normpath(
                                    self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\Surface2Bump.csv'))
                                indx = int(recipe_list[i])

                            img = cv2.imread(os.path.normpath(root + '\\ADC\\' + file))
                            avg_pixel = cv2.mean(img)[0]
                            if avg_pixel > 10:
                                shutil.copyfile(os.path.normpath(root + '\\ADC\\' + file),
                                    os.path.normpath(self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\' + file))
                        except:
                            break

    def scrollbar(self):
        try:
            list = self.f_object[0]
            file_indx = self.f_object[1]
            self.current_image_path = list[file_indx]
            self.image_changing(self.current_image_path)
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            geo = self.label_5.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_5.setPixmap(pixmap)
        except:
            x=1
    #select source folder and pre requsits

    def select_s(self):
        self.Source_listWidget.clear()
        root = Tk()
        root.withdraw()
        self.Source_path=filedialog.askdirectory(title="select folder")
        self.Source_TextEdit.setPlainText(self.Source_path)
        self.files_list = self.get_image_list_from_root(self.Source_path)
        self.Source_listWidget.addItems(self.files_list)
        temp_img = self.find_first_image(self.Source_path)
        im = Image.open(temp_img)
        if self.tabWidget.currentWidget().objectName() == 'crop_tab':
            pixmap = QPixmap(temp_img)
            self.Source_image_size.setText('X: ' + str(im.width) + '       Y:' + str(im.height))
            pixmap=pixmap.scaled(300,300)
            self.label_image.resize(300, 300)
            self.label_image.setPixmap(pixmap)
        elif self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            self.f_object = [self.files_list,0]
            self.current_image_path = self.files_list[0]
            self.image_changing(self.current_image_path)
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            geo = self.label_5.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_5.setPixmap(pixmap)

    #select destination folder
    def select_d(self):
        path=filedialog.askdirectory(title="select folder")
        self.destination_TextEdit.setPlainText(path)
        self.destination_path=path
        self.destination_listWidget.clear()

    #find first image in root folders
    def find_first_image(self, root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                arr = file.split(".")
                if os.path.normpath(arr[-1].lower()) in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                    img_path = os.path.normpath(root + "/" + file )
                    return img_path

    def get_image_list_from_root(self, root_path):
        files_list=[]
        for root, dirs, files in os.walk(root_path):
            for file in files:
                arr = file.split(".")
                if os.path.normpath(arr[-1].lower()) in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                    files_list.append(os.path.normpath(root + "/" + file ))
        return files_list

    # convert images
    def image_convert_and_crop(self):
        try:
            self.Log_listwidget.addItem("start converting and cropping")
            self.Log_listwidget.scrollToBottom()
            D_path = filedialog.askdirectory(title="select output folder")
            self.destination_listWidget.addItem("Start convertion")
            new_width =int(self.output_X_size.toPlainText())
            new_height =int(self.output_Y_size.toPlainText())
            file_list=os.listdir(self.Source_path)
            self.Log_listwidget.addItem("ploting example of foxed offset image")
            self.Log_listwidget.scrollToBottom()
            if self.radioButton.isChecked():
                for root, dirs, files in os.walk(self.Source_path):
                    for file_ in file_list:
                        tmp_str = file_.split('.')
                        if tmp_str[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                            img = Image.open(self.Source_path + '/' + file_)
                            w,h = img.size
                            img1 = self.Image_convert(math.ceil((w - new_width)/2),math.ceil((h - new_height) / 2),img)
                            img1.save(D_path +'\\' + file_)
            #if self.Image_view.messagebox(os.getcwd() +'/temp.jpeg'):
            else:
                self.Log_listwidget.addItem("staring to convert images")
                for folder in file_list:
                    if os.path.isdir(folder):
                        curr_f=D_path + '/' + folder
                        os.mkdir(curr_f)
                        im_list=os.listdir(self.Source_path + '/' + folder)
                        for im in im_list:
                            format = im.split('.')
                            if format[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                                    img = Image.open(self.Source_path + '/' + folder + '/' + im)
                                    w, h = img.size
                                    if w >= new_width and h >= new_height:
                                        left = math.ceil((w - new_width) / 2)
                                        top = math.ceil((h - new_height) / 2)
                                        right = math.ceil((w + new_width) / 2)
                                        bottom = math.ceil((h + new_height) / 2)
                                        if self.radioButton.isChecked():
                                            img1 = self.Image_convert( left ,top ,img)
                                        else:
                                            img1 = img.crop((left, top, right, bottom))
                                        x=im.split('.')
                                        new_im_name=''
                                        new_im_name=new_im_name.join(x[:-1])
                                        try:
                                            img1.save(curr_f + '/' + new_im_name + '.jpeg')
                                        except:
                                            self.log_listWidget.addItem("An exception occurred")
                    else:
                        format = folder.split('.')
                        if format[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                            img = Image.open(self.Source_path + '/' + folder)
                            w, h = img.size
                            if w >= new_width and h >= new_height:
                                left = math.ceil((w - new_width) / 2)
                                top = math.ceil((h - new_height) / 2)
                                right = math.ceil((w + new_width) / 2)
                                bottom = math.ceil((h + new_height) / 2)
                                if self.radioButton.isChecked():
                                    img1 = self.Image_convert(left, top, img)
                                img1 = img.crop((left, top, right, bottom))
                                new_im_name = ''
                                new_im_name = new_im_name.join(format[:-1])
                                try:
                                    img1.save(D_path + '/' + new_im_name + '.jpeg')
                                except:
                                    self.log_listWidget.addItem("An exception occurred")
            self.Log_listwidget.addItem("finish convert")
            self.Log_listwidget.scrollToBottom()
        except:
            messagebox.showinfo(title='Error massage', message='no size')

    def Image_convert(self ,left ,top ,img):
        w,h = img.size
        x_offset = int(self.offset_x.toPlainText())
        y_offset = int(self.offset_y.toPlainText())
        img1 = Image.new(img.mode, (w + x_offset, h + y_offset))
        img1.paste(img, (left - x_offset, top - x_offset))
        return img1

    #write duplicates in table GUI
    def Add_items_to_table(self, duplicate_paths):
        depth=0
        for r in duplicate_paths:
            d=len(r)
            if d > depth:
                depth = d

        l = len(duplicate_paths)
        self.tableWidget.setRowCount(l)
        self.tableWidget.setColumnCount(depth)
        header=[]
        for i in range(depth):
            self.tableWidget.setColumnWidth(i, 450)
            header.append("path " + str(i))
            #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.setHorizontalHeaderLabels(header)
        for n, row_ in enumerate(duplicate_paths):
            for m, str_tmp in enumerate(row_):
                str_tmp = QtWidgets.QTableWidgetItem(row_[m])
                self.tableWidget.setItem(n,m,str_tmp)
        self.tableWidget.setSortingEnabled(1)

    def find_duplicates(self):
        # Create a dictionary to store image hashes and their corresponding file paths
        self.Log_listwidget.addItem("start searching")
        self.Log_listwidget.scrollToBottom()
        image_hashes = {}
        image_hashes2 = {}
        root = Tk()
        root.withdraw()
        root_folder = filedialog.askdirectory(title="select folder")
        widgets = ['Loading: ', progressbar.AnimatedMarker()]
        bar = progressbar.ProgressBar(widgets=widgets).start()
        self.progressBar.setValue(0)
        len_dirs = 0
        if not self.checkBox_2.isChecked():
            # Walk through all files in the root folder and its subfolders
            self.Log_listwidget.addItem("Walk through all files in the root folder and its subfolders")
            self.Log_listwidget.scrollToBottom()

            obj = os.listdir(root_folder)
            indx = 1
            for entry in obj:
                if os.path.isdir(root_folder + '\\' + entry ) :
                    indx = indx +1
            factor = 100/indx
            offset = 0
            tmp_root = root_folder
            for root, dirs, files in os.walk(root_folder):
                len_files=len(files)
                if len_dirs == 0:
                    len_dirs=len(dirs)+1
                for p, file in enumerate(files):
                    # Get the file path
                    file_path = os.path.join(root, file)
                    if file_path.endswith(".jpeg"):
                        # Calculate the file's SHA-1 hash
                        sha1 = hashlib.sha1()
                        with open(file_path, 'rb') as f:
                            while True:
                                data = f.read(1024)
                                if not data:
                                    break
                                sha1.update(data)
                        file_hash = sha1.hexdigest()

                        # If the hash already exists in the dictionary, it's a duplicate
                        if file_hash in image_hashes:
                            image_hashes[file_hash].append(file_path)
                            str = file_path.split("\\")
                            image_hashes2[file_hash].append(str[-2] + '\\' + str[-1] )
                        else:
                            image_hashes[file_hash] = [file_path]
                            str = file_path.split("\\")
                            image_hashes2[file_hash] = [str[-2] + '\\' + str[-1]]

                    if tmp_root != root:
                        offset = offset + factor
                        tmp_root = root
                    precentage = int(((p + 1) / len_files) * factor) + offset
                    self.progressBar.setValue(precentage)
            self.progressBar.setValue(100)
            # Create a list of duplicate images
            duplicates = []
            self.sub_window.duplicates2 = []
            self.Log_listwidget.addItem("Comparing duplicate images")
            self.Log_listwidget.scrollToBottom()
            for file_hash, file_paths in image_hashes2.items():
                if len(file_paths) > 1:
                    x=1
                    duplicates.append(file_paths)
                    self.sub_window.duplicates2.append(image_hashes[file_hash])
            self.Add_items_to_table(duplicates)
            self.Log_listwidget.addItem("finish comparing")
            self.Log_listwidget.scrollToBottom()
        else:
            img_lst = []
            med_lst = []
            for root, dirs, files in os.walk(root_folder):
                len_files = len(files)
                if len_dirs == 0:
                    len_dirs = len(dirs) + 1
                indx = 1
                for p, file in enumerate(files):
                    # Get the file path
                    file_path = os.path.join(root, file)
                    if file_path.endswith(".jpeg"):
                        img = cv2.imread(file_path, 0)
                        array_vec = np.array(img)
                        med_hist = np.median(array_vec)
                        img_lst.append(file_path)
                        med_lst.append(str(med_hist))
                    self.progressBar.setValue(int((p / len_files) * 80))
            arr = np.array([med_lst,img_lst]).T
            df = pd.DataFrame(arr, columns=['hist_med', 'img path'])
            df.sort_values('hist_med')
            sorted_arr =df.sort_values('hist_med')
            self.val_list = df['hist_med'].astype(float)
            self.path_list = df['img path']
            self.candidate_list=[]

            self.horizontalSlider.setMinimum(int(self.val_list.min())/2.55)
            self.horizontalSlider.setMaximum(int(self.val_list.max())/2.55 + 1)
            self.horizontalSlider_2.setMinimum(int(self.val_list.min())/2.55)
            self.horizontalSlider_2.setMaximum(int(self.val_list.max())/2.55 + 1)
            self.hist_max.setText('max:' + self.val_list.max())
            self.hist_min.setText('min:' + self.val_list.min())

    def lower_th_display(self):
        outlier_list=[]
        self.hist_value.setText('value:' + str(int(self.horizontalSlider.value()*2.55)))
        if self.horizontalSlider.value()>self.horizontalSlider_2.value():
            self.horizontalSlider_2.setValue(self.horizontalSlider.value()+1)
        th = int(self.horizontalSlider.value() * 2.55)
        th2 = int(self.horizontalSlider_2.value() * 2.55)
        l = len(self.val_list)
        for m in range(l):
            if float(self.val_list[m]) < th:
                outlier_list.append(self.path_list[m])
            if float(self.val_list[m]) > th2:
                outlier_list.append(self.path_list[m])
        out_len = len(outlier_list)
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(out_len)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['path'])
        for i,path in enumerate(outlier_list):
            self.tableWidget.setItem(i-1,1,QtWidgets.QTableWidgetItem(path))


            #self.hist_value_2.setText(str(int(self.hist_value.text() +1)))

    def upper_th_display(self):
        outlier_list = []
        self.hist_value_2.setText('value:' + str(int(self.horizontalSlider_2.value()*2.55)))
        if self.horizontalSlider.value()>self.horizontalSlider_2.value():
            self.horizontalSlider.setValue(self.horizontalSlider_2.value()-1)
        th = int(self.horizontalSlider.value() * 2.55)
        th2 = int(self.horizontalSlider_2.value() * 2.55)
        l = len(self.val_list)
        for m in range(l):
            if int(float(self.val_list[m])) < th:
                outlier_list.append(self.path_list[m])
            if int(float(self.val_list[m])) > th2:
                outlier_list.append(self.path_list[m])
        out_len = len(outlier_list)
        self.tableWidget.clearContents()
        self.tableWidget.setRowCount(out_len)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setHorizontalHeaderLabels(['path'])
        for i, path in enumerate(outlier_list):
            self.tableWidget.setItem(i - 1, 1, QtWidgets.QTableWidgetItem(path))

        scene = QGraphicsScene()
        self.graphicsView.setScene(scene)
        it = ValueItem(5)
        scene.addItem(it)
        #view = self.graphicsView(scene)
        #view.show()

    def derive_hist_imges(self):
        x=1

    def table_clicked(self):

        self.sub_window.c = self.tableWidget.currentColumn()
        self.sub_window.r = self.tableWidget.currentRow()
        self.sub_window.show()
        self.sub_window.column_max = self.tableWidget.columnCount()
        if self.sub_window.c == 0:
            self.sub_window.Button_prev.hide()
            self.sub_window.Button_next.show()
        elif self.sub_window.c +1 ==self.sub_window.column_max:
            self.sub_window.Button_next.hide()
            self.sub_window.Button_prev.show()
        else:
            self.sub_window.Button_prev.show()
            self.sub_window.Button_next.show()
        self.sub_window.update_image(os.path.normpath(self.sub_window.duplicates2[self.sub_window.r][self.sub_window.c]))

    def next_image(self):
        x=1

#Init app
if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win=UI()
    main_win.show()
    app.exec_()