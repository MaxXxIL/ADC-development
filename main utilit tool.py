# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import time
import math
from PyQt5.QtWidgets import QApplication ,QMainWindow ,QPushButton ,QWidget ,QListWidget ,QLabel ,QListView ,QMessageBox
from PyQt5.QtGui import QPixmap
from Image_utility_tool import Ui_MainWindow
from Image_plot import Ui_plot_image
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import sys
import shutil
import os
import hashlib
import progressbar
from PyQt5.QtGui import QPixmap, QPalette
from tqdm import tqdm
import csv
import numpy as np

from tkinter import messagebox

from PIL import Image , ImageDraw
import PyQt5.QtGui as QG


from tkinter import filedialog
from tkinter import *
from PIL import Image
from easygui import *
class page(Ui_plot_image, QMainWindow):
    def __init__(self,table):
        super().__init__()
        self.setupUi(self)
        self.table = table
        self.Button_next.clicked.connect(self.next_image)
        self.Button_prev.clicked.connect(self.prev_image)
        self.Button_del.clicked.connect(self.delete_image)
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
            self.f_path = self.table.item(self.r, self.c).text()
            geo = self.label.geometry().getRect()
            pixmap = QPixmap(self.f_path)
            pixmap_resized = pixmap.scaled(geo[3], geo[3])
            self.label_2.setText("Image path: " + self.f_path)
            self.label.setPixmap(pixmap_resized)
            self.label.setScaledContents(True)
        except:
            messagebox.showinfo(title='Error massage', message='empty cell')


    def delete_image(self):
        try:
            os.remove(self.f_path)
        except:
            messagebox.showinfo(title='Error massage', message='Image is not exist')

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
            self.f_path = self.table.item(self.r, self.c).text()
            geo = self.label.geometry().getRect()
            pixmap = QPixmap(self.f_path)
            pixmap_resized = pixmap.scaled(geo[3], geo[3])
            self.label_2.setText("Image path: " + self.f_path)
            self.label.setPixmap(pixmap_resized)
            self.label.setScaledContents(True)
        except:
            messagebox.showinfo(title='Error massage', message='empty cell')


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



    # mouse clicked on image
    def mousePressEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            x = 1
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
                # complete saving and next image ------------------- to dooooo




    # initialize all functions
    def init_button_actions(self):
        #low_rez = QtCore.QSize(40, 40)
        self.setWindowTitle("ADC utility tool")
        # pixmap = QPixmap('bb.jpg')
        # pixmap.scaled(1171, 742)
        # self.setStyleSheet(f'background-image: url({pixmap.toImage()}); background-position: right; background-repeat: repeat;')

        self.offset_y.hide()
        self.offset_x.hide()
        self.x_offset_text.hide()
        self.y_offset_text.hide()
        self.radioButton.clicked.connect(self.offset_set)
        self.source_button.clicked.connect(self.select_s)
        self.destination_button.clicked.connect(self.select_d)
        self.commandLinkButton.clicked.connect(self.image_convert_and_crop)
        self.Star_seperate.clicked.connect(self.seperate_images)
        self.seek_identical.clicked.connect(self.find_duplicates)
        self.tableWidget.clicked.connect(self.table_clicked)
        self.Extractor_next.clicked.connect(self.next_file_in_list)
        self.Extractor_prev.clicked.connect(self.prev_file_in_list)

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
        self.current_image_path = list[file_indx]
        self.f_object[1] = file_indx
        l = len(list)
        self.Log_listwidget.addItem("next image " + str(file_indx) + "/" + str(l))
        self.Log_listwidget.scrollToBottom()
        if l - 1 == file_indx :
            self.Extractor_next.hide()
        pixmap = QPixmap(self.current_image_path)
        geo = self.label_5.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_5.setPixmap(pixmap)

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
        pixmap = QPixmap(self.current_image_path)
        geo = self.label_5.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_5.setPixmap(pixmap)

    #seperate images into subfolders
    def seperate_images(self):
        widgets = ['Loading: ', progressbar.AnimatedMarker()]
        bar = progressbar.ProgressBar(widgets=widgets).start()
        self.progressBar.setValue(0)
        N = int(self.plainTextEdit.toPlainText())  # number of images per subfolder
        self.Log_listwidget.addItem("start seperation into subfolders - each folder contains: " + str(N))
        self.Log_listwidget.scrollToBottom()
        path = self.Source_path

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
            pixmap = QPixmap(self.current_image_path)
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
                        pp = file_
                        break
            img = Image.open(self.Source_path + '/' + pp)
            w,h = img.size
            img1 = self.Image_convert(math.ceil((w - new_width)/2),math.ceil((h - new_height) / 2),img)
            img1.save(os.getcwd() +'/temp.jpeg')
        if self.Image_view.messagebox(os.getcwd() +'/temp.jpeg'):
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
        root = Tk()
        root.withdraw()
        root_folder = filedialog.askdirectory(title="select folder")
        widgets = ['Loading: ', progressbar.AnimatedMarker()]
        bar = progressbar.ProgressBar(widgets=widgets).start()
        self.progressBar.setValue(0)
        len_dirs=0
        # Walk through all files in the root folder and its subfolders
        self.Log_listwidget.addItem("Walk through all files in the root folder and its subfolders")
        self.Log_listwidget.scrollToBottom()
        for root, dirs, files in os.walk(root_folder):
            len_files=len(files)
            if len_dirs == 0:
                len_dirs=len(dirs)+1
            indx=1
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
                    else:
                        image_hashes[file_hash] = [file_path]
                self.progressBar.setValue(int((p/len_files)*80))
            indx=indx+1
            self.progressBar.setValue(100)
        # Create a list of duplicate images
        duplicates = []
        self.Log_listwidget.addItem("Comparing duplicate images")
        self.Log_listwidget.scrollToBottom()
        for file_hash, file_paths in image_hashes.items():
            if len(file_paths) > 1:
                duplicates.append(file_paths)
        self.Add_items_to_table(duplicates)
        self.Log_listwidget.addItem("finish comparing")
        self.Log_listwidget.scrollToBottom()

    def table_clicked(self):

        self.sub_window.c = self.tableWidget.currentColumn()
        self.sub_window.r = self.tableWidget.currentRow()

        self.f_path = self.tableWidget.currentItem().text()
        self.sub_window.label_2.setText("Image path: " + self.f_path)
        geo = self.sub_window.label.geometry().getRect()
        pixmap = QPixmap(self.f_path)
        pixmap_resized = pixmap.scaled(geo[3],geo[3])
        self.sub_window.show()
        self.sub_window.f_path = self.f_path
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
        self.sub_window.label.setPixmap(pixmap_resized)
        self.label.setScaledContents(True)
        #w.label.setPixmap(pixmap)
        #im=Image.open(f_path)
        #im.show()

    def next_image(self):
        x=1

#Init app
if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win=UI()
    main_win.show()
    app.exec_()