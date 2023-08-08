# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
from datetime import datetime
import math
import sys
import shutil
import os
import hashlib
import pandas as pd
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QListWidgetItem ,QVBoxLayout
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot, QMetaObject, pyqtSignal
from Image_utility_tool import Ui_MainWindow
from Image_plot import page
from tkinter import messagebox, filedialog
import tkinter as tk
from PIL import Image, ImageDraw
import time

class SIFT_Worker(QThread):
    comparison_done = pyqtSignal(QListWidgetItem)  # Signal to indicate the comparison is done
    found_similar_group = pyqtSignal(dict)
    precentage = pyqtSignal(float)
    delta = pyqtSignal(str)
    def __init__(self, folder_path,th,roi):
        super(SIFT_Worker, self).__init__()
        self.folder_path = folder_path
        self.ROI = roi
        self.th = th
        self.results = []

    def run(self):
        groups = []
        similar_images_dict = {}
        #loading images from path to dict
        self.no_featur_list =[]
        image_dict = self.load_images_to_dict(self.folder_path)
        self.Print_log_item(" ".join(["start comparing : ", str(len(image_dict)), " files"]), "black")

        while len(image_dict) > 1:
            similar_group = []
            #importing images and path from dict
            path_list = list(image_dict.keys())
            image_list = list(image_dict.values())

            #using first image for compare
            img1 = image_list[0]
            img_name = path_list[0]
            self.Ref_image_path = img_name
            del image_dict[path_list[0]]

            #finding similar group
            #t_clock1 = datetime.now()
            similar_group = self.SIFT_Akaze_algo(img1, image_dict, groups, similar_group)
            for key in self.no_featur_list:
                if key in image_dict:
                    del image_dict[key]
            if similar_group is None:
                continue
            similar_images_len = len(similar_group[0])
            if similar_images_len > 0:
                # Add the group of similar images dict and geoups
                similar_images_dict , groups = self.append_group(groups,similar_group,img_name,similar_images_dict)
                #earsing the images that found to be similar
                for key in similar_group[0][:]:
                    del image_dict[key]
                file_name = path_list[0].split('\\')
                self.Print_log_item(" ".join(["Found: ", str(similar_images_len), " images that are similar to file", file_name[-1]]),"blue")

                #emit similar group to gui
                self.found_similar_group.emit(similar_images_dict)
            self.Print_log_item(" ".join(["there is : ", str(len(image_dict)), " images left"]) ,"black")
        similar_images_dict["no_feature"] = self.no_featur_list
        self.found_similar_group.emit(similar_images_dict)
        self.Print_log_item(" ".join(["Comparison done, found: ", str(len(similar_images_dict)), " groups of similar images"]), "green")
        self.found_similar_group.emit(similar_images_dict)

    def load_images_to_dict(self,root_folder):
        image_dict = {}
        file_list = os.listdir(root_folder)
        for i,file in enumerate(file_list):
            self.precentage.emit(i/len(file_list)*100)
            if file.lower().endswith(('.jpeg', '.jpg')):
                image_path = os.path.normpath(os.path.join(root_folder, file))
                image = cv2.imread(image_path)
                width, hight, bands = image.shape
                x1 = int(width / 2 - 200)
                x2 = int(width / 2 + 200)
                y1 = int(hight / 2 - 200)
                y2 = int(hight / 2 + 200)
                cropped = image[x1:x2,y1:y2]
                resized = cv2.resize(cropped, (224,224), interpolation=cv2.INTER_AREA)
                image_dict[image_path] = image
        self.precentage.emit(100)
        return image_dict

    def SIFT_Akaze_algo(self,image,img_dict,groups,similar_group):
        correlation_values = []
        akaze_params = {
            'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # You can adjust this based on your needs
            'descriptor_size': 0,  # The default value (0) uses the optimal size for the descriptor type
            'descriptor_channels': image.shape[-1],
            'threshold': 0.001,  # Adjust this threshold to control the number of keypoints detected
            'nOctaves': 4,  # Number of octaves in the scale-space pyramid
            'nOctaveLayers': 4,  # Number of layers per octave
            'diffusivity': cv2.KAZE_DIFF_PM_G1  # Choose the diffusivity type
        }
        akaze = cv2.AKAZE_create(**akaze_params)
        th = self.th

        croped_image_ref = self.resize_image_arr(image)
        keypoints_ref, descriptors_ref = akaze.detectAndCompute(croped_image_ref, None)
        if descriptors_ref is None or descriptors_ref.shape[0] == 0:
            self.Print_log_item("empty Descriptors for: " + self.Ref_image_path,"red")
            self.no_featur_list.append(self.Ref_image_path)
            return None
        for img in img_dict.values():
            # Detect keypoints and compute descriptors for reference and target images
            croped_image_tar = self.resize_image_arr(img)
            keypoints_target, descriptors_target = akaze.detectAndCompute(croped_image_tar, None)
            if descriptors_target is None or descriptors_target.shape[0] == 0:

                image_key = self.get_key_by_value(img_dict, img)
                if image_key not in self.no_featur_list:
                    self.Print_log_item("empty Descriptors for: " + image_key,"red")
                    self.no_featur_list.append(image_key)
                continue  # Skip this iteration and proceed to the next image

            # Perform AKAZE matching on the keypoints and descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors_ref, descriptors_target)
            # Apply ratio test to filter good matches


            good_matches = [match for match in matches if match.distance < 0.5 * np.max([m.distance for m in matches])]
            num_matches = len(good_matches)
            num_correct_matches = sum(1 for match in good_matches if match.distance < 0.5)
            match_ratio = num_correct_matches / num_matches if num_matches > 0 else 0.0
            keypoints_matched_ratio = len(good_matches) / len(keypoints_ref) if len(keypoints_ref) > 0 else 0.0

            image_key = self.get_key_by_value(img_dict,img)
            already_in_group = any(
                os.path.normpath(os.path.join(self.folder_path, image_key)) in group for group in groups)

            if not already_in_group:
                # Add similar image to the group
                if keypoints_matched_ratio > th:
                    similar_group.append(os.path.normpath(os.path.join(self.folder_path, image_key)))
                    correlation_values.append(str(keypoints_matched_ratio))


        res = [similar_group,correlation_values]
        return (res)

    def Print_log_item(self,str,color):
        item = QListWidgetItem(str)
        item.setForeground(QColor(color))  # Set the font color to red
        self.comparison_done.emit(item)
        time.sleep(0.0001)
        return item

    def append_group(self,groups,similar_group,img_name,dict):
        groups.append(similar_group[0][:])
        arr = np.array([similar_group[0][:], similar_group[1]]).T
        self.df = pd.DataFrame(arr, columns=['Images', 'correlation'])
        images_in_group = len(arr)
        img_str  = img_name.split("\\")[-1]
        dict[str(images_in_group) + " images in group: " + img_str] = self.df
        return dict,groups

    def get_key_by_value(self,dict, value):
        for key, val in dict.items():
            if np.array_equal(val, value):
                return key
        # If the value is not found, you can handle the situation accordingly.
        # For example, you can return None or raise an exception.
        return None

    def resize_image_arr(self,arr):
        geo = arr.shape
        mid_x = int(geo[0] / 2)
        mid_y = int(geo[1] / 2)
        x1 = mid_x - self.ROI
        x2 = mid_x + self.ROI
        y1 = mid_y - self.ROI
        y2 = mid_y + self.ROI
        image_arr = arr[x1:x2, y1:y2]
        return image_arr
class UI(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        layout = QVBoxLayout(self.widget)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)


        self.init_button_actions()
        self.sub_window = page(self.tableWidget)
        self.sub_window.Main_window = self

#------------- initialize all clickes ----------------------
    def init_button_actions(self):
        self.radioButton.clicked.connect(self.checkBox_set_offset)
        self.delete_group.clicked.connect(self.delete_similar_group)
        self.source_button.clicked.connect(self.select_folder)
        self.destination_button.clicked.connect(self.select_folder)
        #self.commandLinkButton.clicked.connect(self.image_convert_and_crop)
        self.Star_seperate.clicked.connect(self.seperate_images)
        self.seek_identical.clicked.connect(self.start_analysis)
        self.tableWidget.clicked.connect(self.table_clicked)
        self.Extractor_next.clicked.connect(self.next_file_in_list)
        self.Extractor_prev.clicked.connect(self.prev_file_in_list)
        self.hist_analysis.clicked.connect(self.checkbox_histogram_view)
        self.horizontalSlider.valueChanged.connect(self.Scrollbar_lower_th_display)
        self.horizontalSlider_2.valueChanged.connect(self.Scrollbar_upper_th_display)
        self.horizontalScrollBar.valueChanged.connect(self.Event_scrollbar)
        self.similar_analysis.clicked.connect(self.checkbox_similar_images)
        self.similar_groups.activated.connect(self.display_comboBox_group)
        self.similar_delete_all.clicked.connect(self.delete_similar_groups)
        self.Crop_checkbox.clicked.connect(self.Checkbox_crop)
        self.seperate_checkbox.clicked.connect(self.Checkbox_seperate)
        self.spinBox_ROI.valueChanged.connect(self.draw_ROI)
        self.init_appereance()

#------------------------ init appereance GUI -----------------------
    def init_appereance(self):
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
        self.similar_groups.hide()
        self.similar_delete_all.hide()
        self.delete_group.hide()
        self.label_9.hide()
        self.label_3.hide()
        self.plainTextEdit.hide()
        self.Recipe_seperate.hide()
        self.doubleSpinBox.hide()
        self.doubleSpinBox.hide()
        self.spinBox_ROI.hide()
        self.label_12.hide()
        self.Source_path=""
        self.destination_path=""

#-----------------------------------------checkbox-----------------------------------------------------
    def checkbox_similar_images(self):
        if self.similar_analysis.isChecked():
            self.similar_groups.show()
            self.hist_analysis.setChecked(False)
            self.sub_window.hist = 0
            self.sub_window.similar_analysis = 1
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

            self.label_9.show()
            self.label_12.show()
            self.doubleSpinBox.show()
            self.spinBox_ROI.show()
        else:
            self.sub_window.similar_analysis = 0
            self.tableWidget.setGeometry(QtCore.QRect(20, 70, 1100, 591))
            self.similar_groups.hide()
            self.label_9.hide()
            self.label_12.hide()
            self.doubleSpinBox.hide()
            self.spinBox_ROI.hide()
            self.delete_group.hide()

    #show histogram objects
    def checkbox_histogram_view(self):
        if self.hist_analysis.isChecked():
            self.sub_window.hist = 1
            self.sub_window.similar_analysis = 0
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

    def Checkbox_crop(self):
        self.Crop_seperate_toggle(self.Crop_checkbox.isChecked())

    def Checkbox_seperate(self):
        self.Crop_seperate_toggle(self.Crop_checkbox.isChecked())

#---------------------------------------------Push buttons-------------------------------------------------------------
    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="select folder")
        sender = self.sender().objectName()
        self.root_folder = folder_path
        if sender == "source_button":
            self.Source_path = folder_path
            self.Source_TextEdit.setPlainText(folder_path)
            self.files_list = self.get_image_list_from_root(self.Source_path)
            temp_img = self.find_first_image(self.Source_path)
            im = Image.open(temp_img)
            self.image_hist_plot(im)
            if self.tabWidget.currentWidget().objectName() == 'crop_tab':
                pixmap = QPixmap(temp_img)
                self.Source_image_size.setText('X: ' + str(im.width) + '       Y:' + str(im.height))
                geo = self.label_image.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_image.setPixmap(pixmap)
            elif self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
                self.f_object = [self.files_list, 0]
                self.current_image_path = self.files_list[0]
                self.image_changing(self.current_image_path)
                pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
                geo = self.label_5.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_5.setPixmap(pixmap)
            else:
                #get ROI
                self.curr_im = im
                self.draw_ROI()

        else:
            self.destination_TextEdit.setPlainText(folder_path)
            self.destination_path = folder_path


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
            im = Image.open(self.current_image_path)
            self.image_hist_plot(im)
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

    def delete_similar_group(self):
        xx= self.similar_groups.currentText()
        for img in list(self.candidates[xx]['Images']):
            os.remove(img)

    def delete_similar_groups(self):
        for group in self.candidates.keys():
            for image in self.candidates[group]["Images"]:
                os.remove(image)

# ------------------------------ GUI Updates---------------
    def image_hist_plot(self,im):
        bands = len(im.getbands())
        arr = np.array(im,dtype=object)
        colors = ["red","green","blue"]
        hist_list = []
        if bands == 1:
            hist_list.append(np.histogram(arr, bins=256, range=(0, 256)))
        else:
            hist_list.append(np.histogram(arr[:,:,0], bins=256, range=(0, 256)))
            hist_list.append(np.histogram(arr[:,:,1], bins=256, range=(0, 256)))
            hist_list.append(np.histogram(arr[:,:,2], bins=256, range=(0, 256)))
        self.plot_widget.clear()
        RGB_string = ""
        if len(hist_list) == 1:
            colors = ["black"]
        for i,channel_hist in enumerate(hist_list):
            counts, bins = channel_hist
            max_indices = np.argpartition(channel_hist[0], -2)[-2:]
            #indx = np.argmax(channel_hist[0])
            RGB_string = RGB_string + colors[i] + ": Median=" + str(max_indices[0]) + " ,"
            try:
                bins_adjusted = bins[:]
                counts = counts.astype(np.float32)
                bins_adjusted = bins_adjusted.astype(np.float32)
                self.plot_widget.plot(bins_adjusted, counts, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150),
                                      pen=colors[i])
                #self.plot_widget.addItem(pg.InfiniteLine(pos=max_indices[0], angle=90, pen=colors[i]))

            except Exception as e:
                print(e)

            self.plot_widget.addItem(pg.InfiniteLine(pos=max_indices[0], angle=90, pen=colors[i]))

            #plt.plot(bins[:-1], counts, color=colors[i], label=f"{colors[i].capitalize()}")
            #plt.axvline(x=max_indices[0],linewidth=1, color=colors[i] , linestyle='--')
        self.plot_widget.setLogMode(y=True)
        self.hist_label.setText(RGB_string)

    def display_comboBox_group(self):
        flag = 0
        group = self.similar_groups.currentText()
        if group == "no_feature":
            img_list =  list(self.candidates[group])
            img_str = self.root_folder + "\\" +img_list[0].split("\\")[-1]
        else:
            img_list =  list(self.candidates[group]['Images'])
            img_str = self.root_folder + "\\" + self.similar_groups.currentText().split(" ")[-1]
        self.Add_items_to_table(img_list,flag)

        self.curr_im = Image.open(img_str)
        self.draw_ROI()
        self.sub_window.duplicates2 = img_list
        self.delete_group.show()

    def update_time_in_log(self, t):
        self.write_to_logview("Compare took: " + str(t) + " seconds")

    def update_bar(self, p):
        self.progressBar.setValue(int(p))

    def update_similar_group(self, similar_dict):
        self.similar_delete_all.show()
        self.similar_groups.clear()
        similar_groups = list(similar_dict.keys())
        self.candidates = similar_dict
        self.similar_groups.addItems(similar_groups)

    def update_list_items(self, results):
        # Update the ListView with the list of results
        self.write_to_logview(results)

    def image_changing(self, path):
        zoom = self.horizontalScrollBar.value()
        image_label_size = self.label_5.geometry().getRect()
        i = Image.open(path)
        width, height = i.size
        self.org_width = width
        self.org_length = height
        self.scale = 1
        if width != height:
            padding = int(abs(width - height) / 2)
            i = self.padding_image(path, padding, width, height)
            width, height = i.size
        if zoom < 15:
            self.scale = 1 + 0.25 * (15 - zoom)
            left = int((1 - 1 / self.scale) * (width / 2))
            right = width - left
            top = int((height / 2) * (1 - 1 / self.scale))
            bot = height - top
            i = i.crop((left, top, right, bot))

            i = i.resize((image_label_size[2], image_label_size[3]), Image.LANCZOS)
        draw = ImageDraw.Draw(i)
        middle_frame = i.size[0] / 2
        draw.rectangle(
            [middle_frame - middle_frame / 10, middle_frame - middle_frame / 10, middle_frame + middle_frame / 10,
             middle_frame + middle_frame / 10],
            outline="red", width=6)
        i.save(os.getcwd() + '\\tmp.jpeg')


    def Crop_seperate_toggle(self,crop):
        sender = self.sender().objectName()
        if sender == 'seperate_checkbox':
            if not crop:
                case = 1
            else:
                case = 0
        else:
            if not crop:
                case = 0
            else:
                case = 1
        self.hide_show_prop(case)


    def hide_show_prop(self,case):
        if case:
            self.label_3.hide()
            self.seperate_checkbox.setChecked(0)
            self.Crop_checkbox.setChecked(1)
            self.plainTextEdit.hide()
            self.Recipe_seperate.hide()
            self.label_2.show()
            self.radioButton.show()
            self.text2.show()
            self.output_X_size.show()
            self.output_Y_size.show()
            self.text2_2.show()

        else:
            self.label_3.show()
            self.seperate_checkbox.setChecked(1)
            self.Crop_checkbox.setChecked(0)
            self.plainTextEdit.show()
            self.Recipe_seperate.show()
            self.label_2.hide()
            self.radioButton.hide()
            self.radioButton.setChecked(0)
            self.text2.hide()
            self.output_X_size.hide()
            self.output_Y_size.hide()
            self.text2_2.hide()
            self.x_offset_text.hide()
            self.y_offset_text.hide()
            self.offset_x.hide()
            self.offset_y.hide()

#-------------------------Events---------------------------


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

    # display Vertical line -Lower th
    def Scrollbar_lower_th_display(self):
        outlier_list=[]
        self.hist_value.setText("Value: " + str(self.horizontalSlider.value()))
        self.plot_hist(self.horizontalSlider.value(),self.horizontalSlider_2.value())
        self.find_hist_outliers()

    # display Vertical line -Upper th
    def Scrollbar_upper_th_display(self):
        outlier_list = []
        self.hist_value_2.setText("Value: " + str(self.horizontalSlider_2.value()))
        self.plot_hist(self.horizontalSlider.value(), self.horizontalSlider_2.value())
        self.find_hist_outliers()

    #Zoom scrollbar change image extravtor display zoom
    def Event_scrollbar(self):
                list = self.f_object[0]
                file_indx = self.f_object[1]
                self.current_image_path = list[file_indx]
                self.image_changing(self.current_image_path)
                pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
                geo = self.label_5.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_5.setPixmap(pixmap)

    # mouse clicked on image to cropp by center
    def mousePressEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            x = event.x()  - 2
            y = event.y() - 43
            if not (x < 0 or y < 0 or x > 600 or y > 600):
                if self.destination_TextEdit.toPlainText() == "":
                    messagebox.showinfo(title='Error massage', message='please select destination folder')
                else:
                    img2 =  self.center_and_crop_image(self.label_5.geometry().getRect(),self.current_image_path,event)
                    if not img2 == None:
                        tmp_str = self.current_image_path.split('\\')
                        img_name = tmp_str[-1]
                        while os.path.exists(self.destination_path + '\\_' + img_name) or os.path.exists(self.destination_path + '\\' + img_name):
                            tmp = img_name.split(".jpeg")
                            img_name = tmp[0] + '_' + '.jpeg'
                        img2.save(self.destination_path + '\\_' + img_name)
                        self.write_to_logview(img_name + ' image was saved')
                        self.next_file_in_list()

    # open Image in plot view from table
    def table_clicked(self):

        self.sub_window.r = self.tableWidget.currentRow()
        self.sub_window.c = self.tableWidget.currentColumn()
        self.sub_window.close()
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

#-----------------------calculation and analysis-------------------------

    def start_analysis(self):
        if not self.Source_path == "":
            self.similar_groups.clear()
            self.Log_listwidget.clear()
            self.write_to_logview("loading images before performing analysis")
            #self.label_8.clear()
            self.tableWidget.clear()
            self.tableWidget.setRowCount(0)

            if self.hist_analysis.isChecked():
                self.image_hist_analysis(self.Source_path)
            elif self.similar_analysis.isChecked():
                self.find_similar(self.Source_path)
            else:
                self.find_duplicate(self.Source_path)
        else:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(title='Error massage', message='Please choose source folder')
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

    def find_similar(self,folder_path):
        # Start a new worker thread for image comparison
        worker = SIFT_Worker(folder_path,self.doubleSpinBox.value(),self.spinBox_ROI.value())
        worker.comparison_done.connect(self.update_list_items)
        worker.found_similar_group.connect(self.update_similar_group)
        worker.precentage.connect(self.update_bar)
        worker.delta.connect(self.update_time_in_log)
        worker.start()

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

    #find first image in root folders
    def find_first_image(self, root_path):
        for root, dirs, files in os.walk(root_path):
            for file in files:
                arr = file.split(".")
                if os.path.normpath(arr[-1].lower()) in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                    img_path = os.path.normpath(root + "/" + file )
                    return img_path

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

    def get_image_list_from_root(self, root_path):
        files_list=[]
        for root, dirs, files in os.walk(root_path):
            for file in files:
                arr = file.split(".")
                if os.path.normpath(arr[-1].lower()) in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :
                    files_list.append(os.path.normpath(root + "/" + file ))
        return files_list

    #seperate images into subfolders
    def seperate_images(self):
        self.progressBar.setValue(0)
        root = tk.Tk()
        root.withdraw()
        if not self.Source_path == "":
            if not self.destination_path =="":
                path = self.Source_path
                #Simple seperation without recipe
                if not self.seperate_checkbox.isChecked():
                    if not (self.output_X_size.toPlainText() == "" and self.output_Y_size.toPlainText()  == ""):
                        # Crop and offset
                        self.write_to_logview("start converting and cropping")
                        D_path = self.destination_path
                        file_list = os.listdir(self.Source_path)
                        self.write_to_logview("ploting example of fixed offset image")
                        new_width = int(self.output_X_size.toPlainText())
                        new_height = int(self.output_Y_size.toPlainText())
                        SourcePath_len = len(self.Source_path)
                        for root, dirs, files in os.walk(self.Source_path):
                            for file_ in files:
                                tmp_str = file_.split('.')
                                if tmp_str[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                                    img = Image.open(root + "\\" + file_)
                                    w, h = img.size
                                    destination_Path = os.path.normpath(D_path + root[SourcePath_len:])
                                    is_dir = os.path.isdir(destination_Path)
                                    if is_dir == False:
                                        os.makedirs(destination_Path)
                                    if self.radioButton.isChecked():
                                        img1 = self.Image_convert(math.ceil((w - new_width) / 2),
                                                                  math.ceil((h - new_height) / 2), img)
                                    else:
                                        left = math.ceil((w - new_width) / 2)
                                        top = math.ceil((h - new_height) / 2)
                                        right = math.ceil((w + new_width) / 2)
                                        bottom = math.ceil((h + new_height) / 2)
                                        img1 = img.crop((left, top, right, bottom))
                                    img1.save(destination_Path + '\\' + file_)

                        else:
                            self.write_to_logview("staring to convert images")

                            for folder in file_list:
                                if os.path.isdir(folder):
                                    curr_f = D_path + '/' + folder
                                    os.mkdir(curr_f)
                                    im_list = os.listdir(self.Source_path + '/' + folder)
                                    for im in im_list:
                                        format = im.split('.')
                                        if format[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                                            img = Image.open(self.Source_path + '/' + folder + '/' + im)
                                            w, h = img.size
                                            if w >= new_width and h >= new_height:
                                                left = math.ceil((w - new_width) / 2)
                                                top = math.ceil((h - new_height) / 2)
                                                right = math.ceil((w + new_width) / 2)
                                                bottom = math.ceil((h + new_height) / 2)
                                                if self.radioButton.isChecked():
                                                    img1 = self.Image_convert(left, top, img)
                                                else:
                                                    img1 = img.crop((left, top, right, bottom))
                                                x = im.split('.')
                                                new_im_name = ''
                                                new_im_name = new_im_name.join(x[:-1])
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

                    else:
                        messagebox.showinfo(title='Error massage', message='Please fill in the output size')
                else:
                    if not (self.offset_x.toPlainText()=="" and self.offset_y.toPlainText()==""):
                    # Seperate folders
                        N = int(self.plainTextEdit.toPlainText())  # number of images per subfolder
                        self.write_to_logview("start seperation into subfolders - each folder contains: " + str(N))
                        folder_num = 0
                        # create the first folder0
                        folder_path = os.path.join(self.destination_path, "folder" + str(folder_num))
                        os.makedirs(folder_path)
                        # loop throught all subfolders in path
                        for root, dirs, files in os.walk(path):
                            l = len(files)
                            for i, file in enumerate(files):
                                self.progressBar.setValue(i / l)
                                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                                    # check if folder is full to max images
                                    if len(os.listdir(folder_path)) >= N:
                                        folder_path = os.path.join(self.destination_path, "folder" + str(folder_num))
                                        os.makedirs(folder_path)
                                        self.write_to_logview("subfolder: " + str(folder_num) + "ready")
                                        folder_num += 1
                                    shutil.copy(os.path.join(root, file), os.path.join(folder_path, file))
                        self.write_to_logview("finished seperat into: " + str(folder_num) + "subfolders")
                    else:
                            messagebox.showinfo(title='Error massage', message='Please fill in the offset')
                    self.write_to_logview("finish convert")
            else:
                messagebox.showinfo(title='Error massage', message='destination folder isnt choosen')
        else:
            messagebox.showinfo(title='Error massage', message='Source folder isnt choosen')


#-----------------------Image proccesing ------------------------- ----------------
    #padding image for future use
    def padding_image(self,path,padding,width,height):
        # Define the padding values.
        image =  Image.open(path)
        top, bottom, left, right =  (padding, padding, 0, 0)
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

    def center_and_crop_image(self,img_rect,cuur_path,event):
        x = event.x() - img_rect[0] - 2
        y = event.y() - img_rect[1] - 43
        if not (x < 0 or y < 0 or x > 600 or y > 600):
            img_dim = Image.open(cuur_path).size
            valid_y = (img_dim[0] -img_dim[1])/2
            factor = img_dim[0] / img_rect[3]
            if (self.scale == 1 and (y > valid_y/factor and y < (img_rect[3] - valid_y/factor))) or self.scale != 1 :
                offset_y = ((img_rect[3] / 2) - y) * factor
                offset_x = ((img_rect[3] / 2) - x) * factor
                if self.scale != 1:
                    offset_x = offset_x / self.scale
                    offset_y = offset_y / self.scale
                offset_x = int(offset_x)
                offset_y = int(offset_y)
                img = Image.open(cuur_path)
                img1 = Image.new(img.mode, (img_dim[0] + abs(int(offset_x)), img_dim[1] + abs(int(offset_y))))
                img1.paste(img, (0 + offset_x, 0 + offset_y))
                img2 = img1.crop((0, 0, img_dim[0], img_dim[1]))
            return img2

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

#------------------------ GUI updates ----------------------------------
    #write logs to logview
    def write_to_logview(self, str1):
        self.Log_listwidget.addItem(str1)
        self.Log_listwidget.scrollToBottom()

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
            self.tableWidget.setColumnWidth(0, 700)
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

    def draw_ROI(self):
        im = self.curr_im
        ROI = self.spinBox_ROI.value()
        geo = self.label_8.geometry().getRect()
        resized_im = im.resize([geo[2], geo[3]])
        draw = ImageDraw.Draw(resized_im)
        middle_frame = geo[2] / 2
        draw.rectangle([middle_frame - ROI / 2,
                        middle_frame - ROI / 2,
                        middle_frame + ROI / 2,
                        middle_frame + ROI / 2],
                       outline="blue", width=2)
        resized_im.save(os.getcwd() + '\\tmp.jpeg')
        pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
        self.label_8.setPixmap(pixmap)
        x=1
#Init app
if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win=UI()
    main_win.show()

    app.exec_()