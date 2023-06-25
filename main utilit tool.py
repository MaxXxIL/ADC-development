# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import sys
import shutil
import os
import hashlib
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from Image_utility_tool import Ui_MainWindow
from Image_plot import page
from tkinter import messagebox, filedialog
import tkinter as tk
from PIL import Image, ImageDraw

class UI(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.init_button_actions()
        self.sub_window = page(self.tableWidget)
        self.sub_window.Main_window = self

    # initialize all functions
    def init_button_actions(self):
        self.Table_flag = 0
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
        self.radioButton.clicked.connect(self.checkBox_set_offset)
        self.source_button.clicked.connect(self.select_s)
        self.destination_button.clicked.connect(self.select_d)
        self.commandLinkButton.clicked.connect(self.image_convert_and_crop)
        self.Star_seperate.clicked.connect(self.seperate_images)
        self.seek_identical.clicked.connect(self.start_analysis)
        self.tableWidget.clicked.connect(self.table_clicked)
        self.Extractor_next.clicked.connect(self.next_file_in_list)
        self.Extractor_prev.clicked.connect(self.prev_file_in_list)
        self.hist_analysis.clicked.connect(self.checkbox_histogram_view)
        self.horizontalSlider.valueChanged.connect(self.Scrollbar_lower_th_display)
        self.horizontalSlider_2.valueChanged.connect(self.Scrollbar_upper_th_display)
        self.horizontalScrollBar.valueChanged.connect(self.Event_scrollbar)
        self.scale_list=np.linspace(0.1,1,num=50)
        self.similar_analysis.clicked.connect(self.checkbox_similar_images)
        self.similar_groups.activated.connect(self.comboBox_group)
        self.similar_groups.hide()

#-----------------------------------------Define actions of objects-----------------------------------------------------
    #show combobox objects
    def checkbox_similar_images(self):
        if self.similar_analysis.isChecked():
            self.similar_groups.show()
            self.hist_analysis.setChecked(False)
            self.sub_window.hist = 0
            self.horizontalSlider.hide()
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 700, 591))
            self.seek_identical.setText('Find identical images')
            self.hist_min.hide()
            self.hist_max.hide()
            self.hist_value.hide()
            self.hist_value_2.hide()
            self.hist_upper.hide()
            self.hist_lower.hide()
            self.horizontalSlider_2.hide()
        else:
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 1100, 591))
            self.similar_groups.hide()

    #show histogram objects
    def checkbox_histogram_view(self):
        if self.hist_analysis.isChecked():
            self.sub_window.hist = 1
            self.horizontalSlider.show()
            self.horizontalSlider_2.show()
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 700, 591))
            self.seek_identical.setText('Analyz images backgroung')
            self.hist_min.show()
            self.hist_max.show()
            self.hist_value.show()
            self.hist_value_2.show()
            self.hist_upper.show()
            self.hist_lower.show()
            self.similar_analysis.setChecked(False)
            self.similar_groups.hide()
        else:
            self.sub_window.hist = 0
            self.horizontalSlider.hide()
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 1100, 591))
            self.seek_identical.setText('Find identical images')
            self.hist_min.hide()
            self.hist_max.hide()
            self.hist_value.hide()
            self.hist_value_2.hide()
            self.hist_upper.hide()
            self.hist_lower.hide()
            self.horizontalSlider_2.hide()

    # show offset objects
    def checkBox_set_offset(self):
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


#---------------------------------------------Handle events-------------------------------------------------------------
##############--Generic Tab--##################

    #select source folder
    def select_s(self):
        self.Source_listWidget.clear()
        root = tk.Tk()
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

##############--Image extractor Tab--##################
    #scroll zoom in Image
    def wheelEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            scrollDistance = event.angleDelta().y()
            numDegrees = event.angleDelta() / 8
            numSteps = numDegrees / 15
            if scrollDistance > 0:
                zoom_indx = -1*numSteps.manhattanLength()
            else:
                zoom_indx = numSteps.manhattanLength()
            s_indx = self.horizontalScrollBar.value()
            self.horizontalScrollBar.setValue(s_indx + zoom_indx)

    def Event_scrollbar(self):
                list = self.f_object[0]
                file_indx = self.f_object[1]
                self.current_image_path = list[file_indx]
                self.image_changing(self.current_image_path)
                pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
                geo = self.label_5.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_5.setPixmap(pixmap)

    # mouse clicked on image
    def mousePressEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            if self.destination_TextEdit.toPlainText() == "":
                messagebox.showinfo(title='Error massage', message='please select destination folder')
            else:
                img2 =  self.center_and_crop_image(self.label_5.geometry().getRect(),self.current_image_path,event)
                tmp_str = self.current_image_path.split('\\')
                img_name = tmp_str[-1]
                while os.path.exists(self.destination_path + '\\_' + img_name) or os.path.exists(self.destination_path + '\\' + img_name):
                    tmp = img_name.split(".jpeg")
                    img_name = tmp[0] + '_' + '.jpeg'
                img2.save(self.destination_path + '\\_' + img_name)
                self.write_to_logview(img_name + ' image was saved')
                self.next_file_in_list()
    #show next image in list

    def next_file_in_list(self):

        list = self.f_object[0]
        file_indx = self.f_object[1] + 1
        try:
            self.current_image_path = list[file_indx]
            self.f_object[1] = file_indx
            l = len(list)
            self.write_to_logview("next image " + str(file_indx) + "/" + str(l))
            if l - 1 == file_indx :
                self.Extractor_next.hide()
            self.image_changing(self.current_image_path)
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            geo = self.label_5.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_5.setPixmap(pixmap)
        except:
            QMessageBox.about(self, "info massage", "no more images")

    # show next image in list
    def prev_file_in_list(self):
        self.Extractor_next.show()
        list = self.f_object[0]
        file_indx = self.f_object[1] -1
        self.current_image_path = list[file_indx]
        self.f_object[1] = file_indx
        l = len(list)
        self.write_to_logview("next image " + str(file_indx) + "/" + str(l))
        if file_indx == 0:
            self.Extractor_next.hide()
        self.image_changing(self.current_image_path)
        pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
        geo = self.label_5.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_5.setPixmap(pixmap)

    def table_clicked(self):

        self.sub_window.r = self.tableWidget.currentRow()
        self.sub_window.c = self.tableWidget.currentColumn()
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
        if not self.hist_analysis.isChecked() and not self.similar_analysis.isChecked():
            self.sub_window.update_image(
                os.path.normpath(self.sub_window.duplicates2[self.sub_window.r][self.sub_window.c]))
        else:
            self.sub_window.update_image(
                os.path.normpath(self.sub_window.duplicates2[self.sub_window.r]))

##############--Image finder Tab--##################
    def start_analysis(self):
        self.similar_groups.clear()
        self.label_8.clear()
        self.tableWidget.clear()
        root = tk.Tk()
        root_folder = filedialog.askdirectory(parent=root, title="Select Folder")
        root.withdraw()

        self.Log_listwidget.clear()
        self.write_to_logview("start searching")

        if self.hist_analysis.isChecked():
            self.image_hist_analysis(root_folder)

        elif self.similar_analysis.isChecked():
            self.candidates = self.find_similar_images(root_folder)
            similar_groups = list(self.candidates.keys())
            self.similar_groups.addItems(similar_groups)

        else:
            self.find_duplicate(root_folder)

    def Scrollbar_lower_th_display(self):
        outlier_list=[]
        self.hist_value.setText("Value: " + str(self.horizontalSlider.value()))
        self.plot_hist(self.horizontalSlider.value(),self.horizontalSlider_2.value())
        self.find_hist_outliers()

    def Scrollbar_upper_th_display(self):
        outlier_list = []
        self.hist_value_2.setText("Value: " + str(self.horizontalSlider_2.value()))
        self.plot_hist(self.horizontalSlider.value(), self.horizontalSlider_2.value())
        self.find_hist_outliers()

#------------------------------------------------GUI functions----------------------------------------------------------
    #show image of choosen image and display on label
    def comboBox_group(self):
        flag = 0
        img_list =  list(self.candidates[self.similar_groups.currentText()]['Images'])
        self.Add_items_to_table(img_list,flag)
        pixmap = QPixmap(self.similar_groups.currentText())
        geo = self.label_8.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_8.setPixmap(pixmap)

    #change image in image_extractor view
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

    #padding image for future use
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

    #write logs intu logview
    def write_to_logview(self, str1):
        self.Log_listwidget.addItem(str1)
        self.Log_listwidget.scrollToBottom()

    #coping ADC files
    def Copy_ADC_folder_files(self,root,recipe,local_path,destination_path):
        shutil.copyfile(os.path.normpath(root + '\\ADC\\image_flow.csv'), os.path.normpath(
            destination_path + '\\' + recipe + '\\' + local_path + '\\ADC\\image_flow.csv'))

        shutil.copyfile(os.path.normpath(root + '\\ADC\\image_view.csv'), os.path.normpath(
            destination_path + '\\' + recipe + '\\' + local_path + '\\ADC\\image_view.csv'))

        shutil.copyfile(os.path.normpath(root + '\\ADC\\ManReClassify.ini'), os.path.normpath(
            destination_path + '\\' + recipe + '\\' + local_path + '\\ADC\\ManReClassify.ini'))

        shutil.copyfile(os.path.normpath(root + '\\ADC\\run_details.json'), os.path.normpath(
            destination_path + '\\' + recipe + '\\' + local_path + '\\ADC\\run_details.json'))

        shutil.copyfile(os.path.normpath(root + '\\ADC\\Surface2Bump.csv'), os.path.normpath(
            destination_path + '\\' + recipe + '\\' + local_path + '\\ADC\\Surface2Bump.csv'))

    #Show images by th out of images GL histogram
    def plot_hist(self,th_low,th_high):
        counts, bins = np.histogram(self.val_list)
        plt.clf()
        plt.cla()
        plt.hist(bins[:-1], bins, weights=counts)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of AVG GL')
        plt.axvline(x=th_low, color='r', linestyle='--')
        plt.axvline(x=th_high, color='r', linestyle='--')
        plt.savefig('histogram.png')
        pixmap = QPixmap('histogram.png')
        geo = self.label_7.geometry().getRect()
        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_7.setPixmap(pixmap)

    #write duplicates in table GUI
    def Add_items_to_table(self, duplicate_paths,flag):
        self.tableWidget.setRowCount(0)
        self.tableWidget.clear()
        #self.sub_window.duplicates2 =[]
        if flag == 1:
            len_dup = [len(x) for x in duplicate_paths]
            self.tableWidget.setRowCount(len(duplicate_paths))
            self.tableWidget.setColumnCount(max(len_dup))
            header=[]
            for i in range(max(len_dup)):
                self.tableWidget.setColumnWidth(i, 450)
                header.append("path " + str(i))
            self.tableWidget.setHorizontalHeaderLabels(header)
            for n, row_ in enumerate(duplicate_paths):
                for m, str_tmp in enumerate(row_):
                    str_tmp = QtWidgets.QTableWidgetItem(row_[m])
                    self.tableWidget.setItem(n,m,str_tmp)
        else:
            self.tableWidget.setRowCount(len(duplicate_paths))
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setColumnWidth(0, 450)
            for n, path in enumerate(duplicate_paths):
                str_tmp = QtWidgets.QTableWidgetItem(path)
                self.tableWidget.setItem(n,0,str_tmp)
                #self.sub_window.duplicates2.append(path)
        self.tableWidget.setSortingEnabled(1)

    # display histogram on label
    def Display_hist(self,arr,th_low,th_high):
        arr_max = max(arr)
        arr_min = min(arr)
        self.horizontalSlider_2.setMaximum(arr_max)
        self.horizontalSlider_2.setMinimum(arr_min)
        self.horizontalSlider_2.setValue(arr_max)
        self.hist_min.setText("Min: " + str(arr_min))
        self.hist_max.setText("Max: " + str(arr_max))
        self.horizontalSlider.setMaximum(arr_max)
        self.horizontalSlider.setMinimum(arr_min)
        self.horizontalSlider.setValue(arr_min)
        self.plot_hist(th_low,th_high)


#-----------------------calculation and analysis-------------------------

    #find first image in root folders
    def find_first_image(self, root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                arr = file.split(".")
                if os.path.normpath(arr[-1].lower()) in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                    img_path = os.path.normpath(root + "/" + file )
                    return img_path

    #seperate images into subfolders
    def seperate_images(self):
        self.progressBar.setValue(0)
        path = self.Source_path
        #Simple seperation without recipe
        if not self.checkBox.isChecked():
            try:
                N = int(self.plainTextEdit.toPlainText())  # number of images per subfolder
                self.write_to_logview("start seperation into subfolders - each folder contains: " + str(N))
                folder_num = 0
                #create the first folder0
                folder_path = os.path.join(self.destination_path, "folder" + str(folder_num))
                os.makedirs(folder_path)
                #loop throught all subfolders in path
                for root, dirs, files in os.walk(path):
                    l  = len(files)
                    for i ,file in enumerate(files):
                        self.progressBar.setValue(i/l)
                        if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                            #check if folder is full to max images
                            if len(os.listdir(folder_path)) >= N:
                                folder_path = os.path.join(self.destination_path, "folder" + str(folder_num))
                                os.makedirs(folder_path)
                                self.write_to_logview("subfolder: " + str(folder_num) + "ready")
                                folder_num += 1
                            shutil.copy(os.path.join(root, file), os.path.join(folder_path, file))
                self.write_to_logview("finished seperat into: " + str(folder_num) + "subfolders")
            except:
                QMessageBox.about(self, "Error msg", "please selects input folder")

        # seperation with recipe
        else:
            try:
                l = len(path)+1
                for root, dirs, files in os.walk(path):
                    if 'ADC' in dirs:
                        tmp_str1 = root[l:]
                        tmp_str = tmp_str1.split('\\')
                        dataframe1 = pd.read_csv(root + '\\ADC\\' + 'Surface2Bump.csv')
                        img_path_list =  dataframe1['defect_key'].values
                        recipe_list = dataframe1['Recipe'].values
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
                                    #copy ADC folder files
                                    self.Copy_ADC_folder_files(root,str(recipe_list[i]),tmp_str1,self.destination_path)
                                img = cv2.imread(os.path.normpath(root + '\\ADC\\' + file))
                                avg_pixel = cv2.mean(img)[0]
                                #check if image is not blank
                                if avg_pixel > 10:
                                    shutil.copyfile(os.path.normpath(root + '\\ADC\\' + file),
                                        os.path.normpath(self.destination_path  + '\\' + str(recipe_list[i]) + '\\' + tmp_str1 + '\\ADC\\' + file))
                            except:
                                break
            except:
                QMessageBox.about(self, "Error msg", "please selects output folder")

    #find outliers by th
    def find_hist_outliers(self):
        df = pd.DataFrame(self.val_list, columns=['hist_med'])
        indices1 = list(df.index[df['hist_med'] < self.horizontalSlider.value()].values)
        indices2 = list(df.index[df['hist_med'] > self.horizontalSlider_2.value()].values)
        indices = indices1 + indices2
        outlier_list=[]
        self.sub_window.duplicates2 = []
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setColumnWidth(0, 450)
        for i in range(len(indices)):
            img_path = self.df.iloc[int(indices[i]), 1]
            str_tmp = QtWidgets.QTableWidgetItem(self.df.iloc[int(indices[i]), 1])
            self.sub_window.duplicates2.append(img_path)
            outlier_list.append(str_tmp)
        self.tableWidget.setRowCount(len(outlier_list))
        if outlier_list:
            for j,item in enumerate(outlier_list):
                self.tableWidget.setItem(j, 0, item)
            self.tableWidget.setColumnCount(1)

    def center_and_crop_image(self,img_rect,cuur_path,event):
        x = event.x() - img_rect[0] - 5
        y = event.y() - img_rect[1] - 45
        img_dim = Image.open(cuur_path).size
        x_new = (img_dim[0] / img_rect[2]) * x
        y_new = (img_dim[1] / img_rect[3]) * y
        offset_y = int((img_dim[1] / 2) - y_new)
        offset_x = int((img_dim[0] / 2) - x_new)
        top = math.ceil((img_dim[0]))
        left = math.ceil((img_dim[1]))
        img = Image.open(cuur_path)
        img1 = Image.new(img.mode, (img_dim[0] + abs(offset_x), img_dim[1] + abs(offset_y)))
        img1.paste(img, (0 + offset_x, 0 + offset_y))
        img2 = img1.crop((0, 0, img_dim[0], img_dim[1]))
        return img2

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
            self.write_to_logview("start converting and cropping")

            D_path = filedialog.askdirectory(title="select output folder")
            self.destination_listWidget.addItem("Start convertion")
            file_list=os.listdir(self.Source_path)
            self.write_to_logview("ploting example of foxed offset image")
            new_width = int(self.output_X_size.toPlainText())
            new_height = int(self.output_Y_size.toPlainText())
            SourcePath_len = len(self.Source_path)
            for root, dirs, files in os.walk(self.Source_path):
                for file_ in files:
                    tmp_str = file_.split('.')
                    if tmp_str[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                        img = Image.open(root + "\\" + file_)
                        w,h = img.size
                        destination_Path = os.path.normpath(D_path + root[SourcePath_len:])
                        is_dir = os.path.isdir(destination_Path)
                        if is_dir == False:
                            os.makedirs(destination_Path)
                        if self.radioButton.isChecked():
                            img1 = self.Image_convert(math.ceil((w - new_width)/2),math.ceil((h - new_height) / 2),img)
                        else:
                            left = math.ceil((w - new_width) / 2)
                            top = math.ceil((h - new_height) / 2)
                            right = math.ceil((w + new_width) / 2)
                            bottom = math.ceil((h + new_height) / 2)
                            img1 = img.crop((left, top, right, bottom))
                        img1.save(destination_Path +'\\' + file_)

            else:
                self.write_to_logview("staring to convert images")

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
                                            self.write_to_logview("An exception occurred")
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
                                    self.write_to_logview("An exception occurred")
            self.write_to_logview("finish convert")
        except:
            messagebox.showinfo(title='Error massage', message='no size')

    def Image_convert(self ,left ,top ,img):
        w,h = img.size
        x_offset = 0
        y_offset = 0
        if self.radioButton.isChecked():
            x_offset = int(self.offset_x.toPlainText())
            y_offset = int(self.offset_y.toPlainText())
        img1 = Image.new(img.mode, (w + x_offset, h + y_offset))
        img1.paste(img, (left - x_offset, top - x_offset))
        return img1

    # find identical images
    def find_duplicate(self,root_folder):
        flag = 1
        # Create a dictionary to store image hashes and their corresponding file paths
        image_hashes = {}
        image_hashes2 = {}
        self.sub_window.duplicates2 = []
        self.progressBar.setValue(0)
        len_dirs = 0
        if not self.hist_analysis.isChecked():
            self.write_to_logview("Walk through all files in the root folder and its subfolders")

            obj = os.listdir(root_folder)
            indx = 1
            for entry in obj:
                if os.path.isdir(root_folder + '\\' + entry ) :
                    indx = indx +1
            factor = 100/indx
            offset = 0
            tmp_root = root_folder
            folder_indx = 1
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
                            image_hashes2[file_hash].append(file_path)
                        else:
                            image_hashes[file_hash] = [file_path]
                            image_hashes2[file_hash] = [file_path]

                    if tmp_root != root:
                        offset = offset + factor
                        tmp_root = root
                        str_folder_indx = "{}".format(folder_indx)
                        str_folder_max = "{}".format(indx)
                        tmp_str = "folder: " + str_folder_indx + "/" + str_folder_max
                        self.write_to_logview(tmp_str)
                        folder_indx += 1
                    precentage = int(((p + 1) / len_files) * 100)
                    self.progressBar.setValue(precentage)
            self.progressBar.setValue(100)
            # Create a list of duplicate images

            self.write_to_logview("Comparing duplicate images")
            for file_hash, file_paths in image_hashes2.items():
                if len(file_paths) > 1:
                    self.sub_window.duplicates2.append(image_hashes[file_hash])
            self.Add_items_to_table(self.sub_window.duplicates2,flag)
            self.write_to_logview("finish comparing")

    #find images with same features
    def find_similar_images(self, folder_path):
        self.Log_listwidget.clear()
        self.write_to_logview("start searching")
        image_files = os.listdir(folder_path)
        similar_images_dict = {}
        groups = []
        window_x, window_y, window_w, window_h = (200, 200, 400, 400)
        list_len = len(image_files)
        for i in range(list_len):
            self.write_to_logview("comparing image :" + str(i))
            self.progressBar.setValue(int(100*i/list_len))
            similar_group =[]
            sift = cv2.SIFT_create()
            img1 = cv2.imread(os.path.join(folder_path, image_files[i]), cv2.IMREAD_GRAYSCALE)
            #img1_window = img1[window_y:window_y + window_h, window_x:window_x + window_w]
            if image_files[i].split(".")[-1] == "jpeg":
                #img1= img1[window_y2:window_y2 + window_h2, window_x2:window_x2 + window_w2]
                similar_group.append( os.path.normpath(folder_path + "\\" + image_files[i]))  # Initialize a group with the current image
                ref_img = [image_files[i]]
                correlation_values = []
                #im = Image.open(os.path.join(folder_path, image_files[i]))
                #im.show()
                for j in range(i + 1, len(image_files)):
                    if image_files[j].split(".")[-1]=="jpeg":
                        sift = cv2.SIFT_create()
                        img2 = cv2.imread(os.path.join(folder_path, image_files[j]), cv2.IMREAD_GRAYSCALE)
                        # Detect keypoints and compute descriptors for reference and target images
                        keypoints_ref, descriptors_ref = sift.detectAndCompute(img1, None)
                        keypoints_target, descriptors_target = sift.detectAndCompute(img2, None)

                        # Initialize the FLANN matcher
                        FLANN_INDEX_KDTREE = 1
                        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                        search_params = dict(checks=50)
                        flann = cv2.FlannBasedMatcher(index_params, search_params)

                        # Match keypoints using FLANN matcher
                        matches = flann.knnMatch(descriptors_ref, descriptors_target, k=2)

                        # Apply ratio test to filter good matches
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                        num_matches = len(good_matches)
                        num_correct_matches = sum(1 for match in good_matches if match.distance < 0.5)
                        match_ratio = num_correct_matches / num_matches if num_matches > 0 else 0.0
                        keypoints_matched_ratio = len(good_matches) / len(keypoints_ref) if len(
                            keypoints_ref) > 0 else 0.0
                        already_in_group = False
                        cuur_img = os.path.normpath(folder_path + "\\" + image_files[j])
                        for group in groups:
                            for img in group:
                                if cuur_img == img:
                                    already_in_group = True
                                    break

                        if not already_in_group:
                            # Add similar image to the group
                            if float(keypoints_matched_ratio) > 0.5:
                                similar_group.append(os.path.normpath(folder_path + "\\" + image_files[j]))
                                correlation_values.append(keypoints_matched_ratio)
                            # Add similar image to the group

            if len(similar_group) > 1:
                # Add the group of similar images to the list
                groups.append(similar_group[1:])
                arr = np.array([similar_group[1:], correlation_values]).T
                self.df = pd.DataFrame(arr, columns=['Images', 'correlation'])
                # similar_images.append(similar_group)
                similar_images_dict[similar_group[0]] = self.df
        self.write_to_logview("finish searching")
        self.progressBar.setValue(100 )
        return similar_images_dict

    # calculate hist of images
    def image_hist_analysis(self,root_folder):
        img_lst = []
        med_lst = []
        for root, dirs, files in os.walk(root_folder):
            len_files = len(files)
            if len_files == 0:
                len_dirs = len(dirs) + 1
            indx = 1
            for p, file in enumerate(files):
                # Get the file path
                file_path = os.path.join(root, file)
                if file_path.endswith(".jpeg"):
                    img = cv2.imread(file_path, 0)
                    array_vec = np.array(img)
                    med_hist = int(np.median(array_vec))
                    img_lst.append(file_path)
                    med_lst.append(med_hist)
                self.progressBar.setValue(int((p / len_files) * 80))

        arr = np.array([med_lst, img_lst]).T
        self.df = pd.DataFrame(arr, columns=['hist_med', 'img path'])

        sorted_arr = self.df.sort_values('hist_med')
        self.val_list = sorted_arr['hist_med'].astype(int)
        self.path_list = sorted_arr['img path']

        max_v = max(self.val_list)
        min_v = min(self.val_list)
        self.Display_hist(self.val_list, min_v, max_v)


#Init app
if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win=UI()
    main_win.show()
    app.exec_()