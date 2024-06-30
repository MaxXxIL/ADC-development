# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cupy as cp
import cv2
from sklearn.cluster import DBSCAN
import math
import sys
import shutil
import os
import hashlib

import numpy
import pandas as pd
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore ,QtGui
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QListWidgetItem ,QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
from Image_plot import page
from tkinter import messagebox, filedialog
import tkinter as tk
from PIL import Image, ImageDraw
import time
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1496, 858)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(260, 0, 1221, 741))
        self.tabWidget.setObjectName("tabWidget")
        self.crop_tab = QtWidgets.QWidget()
        self.crop_tab.setObjectName("crop_tab")
        self.label_image = QtWidgets.QLabel(self.crop_tab)
        self.label_image.setGeometry(QtCore.QRect(0, 180, 500, 500))
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.label = QtWidgets.QLabel(self.crop_tab)
        self.label.setGeometry(QtCore.QRect(0, 690, 101, 21))
        self.label.setObjectName("label")
        self.Source_image_size = QtWidgets.QLabel(self.crop_tab)
        self.Source_image_size.setGeometry(QtCore.QRect(110, 690, 171, 21))
        self.Source_image_size.setObjectName("Source_image_size")
        self.text2 = QtWidgets.QLabel(self.crop_tab)
        self.text2.setGeometry(QtCore.QRect(30, 80, 21, 21))
        self.text2.setObjectName("text2")
        self.label_2 = QtWidgets.QLabel(self.crop_tab)
        self.label_2.setGeometry(QtCore.QRect(30, 50, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.text2_2 = QtWidgets.QLabel(self.crop_tab)
        self.text2_2.setGeometry(QtCore.QRect(110, 80, 21, 21))
        self.text2_2.setObjectName("text2_2")
        self.output_X_size = QtWidgets.QTextEdit(self.crop_tab)
        self.output_X_size.setGeometry(QtCore.QRect(50, 80, 51, 31))
        self.output_X_size.setObjectName("output_X_size")
        self.output_Y_size = QtWidgets.QTextEdit(self.crop_tab)
        self.output_Y_size.setGeometry(QtCore.QRect(130, 80, 51, 31))
        self.output_Y_size.setObjectName("output_Y_size")
        self.radioButton = QtWidgets.QRadioButton(self.crop_tab)
        self.radioButton.setGeometry(QtCore.QRect(230, 50, 91, 21))
        self.radioButton.setObjectName("radioButton")
        self.offset_y = QtWidgets.QTextEdit(self.crop_tab)
        self.offset_y.setGeometry(QtCore.QRect(300, 80, 51, 31))
        self.offset_y.setObjectName("offset_y")
        self.y_offset_text = QtWidgets.QLabel(self.crop_tab)
        self.y_offset_text.setGeometry(QtCore.QRect(280, 80, 31, 20))
        self.y_offset_text.setObjectName("y_offset_text")
        self.offset_x = QtWidgets.QTextEdit(self.crop_tab)
        self.offset_x.setGeometry(QtCore.QRect(220, 80, 51, 31))
        self.offset_x.setObjectName("offset_x")
        self.x_offset_text = QtWidgets.QLabel(self.crop_tab)
        self.x_offset_text.setGeometry(QtCore.QRect(200, 80, 21, 21))
        self.x_offset_text.setObjectName("x_offset_text")
        self.Star_seperate = QtWidgets.QPushButton(self.crop_tab)
        self.Star_seperate.setGeometry(QtCore.QRect(30, 150, 141, 23))
        self.Star_seperate.setObjectName("Star_seperate")
        self.Recipe_seperate = QtWidgets.QCheckBox(self.crop_tab)
        self.Recipe_seperate.setGeometry(QtCore.QRect(30, 110, 131, 17))
        self.Recipe_seperate.setObjectName("Recipe_seperate")
        self.label_3 = QtWidgets.QLabel(self.crop_tab)
        self.label_3.setGeometry(QtCore.QRect(30, 50, 151, 21))
        self.label_3.setObjectName("label_3")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.crop_tab)
        self.plainTextEdit.setGeometry(QtCore.QRect(30, 80, 101, 21))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.Crop_checkbox = QtWidgets.QCheckBox(self.crop_tab)
        self.Crop_checkbox.setEnabled(True)
        self.Crop_checkbox.setGeometry(QtCore.QRect(30, 20, 101, 18))
        self.Crop_checkbox.setChecked(True)
        self.Crop_checkbox.setObjectName("Crop_checkbox")
        self.seperate_checkbox = QtWidgets.QCheckBox(self.crop_tab)
        self.seperate_checkbox.setGeometry(QtCore.QRect(150, 20, 141, 18))
        self.seperate_checkbox.setObjectName("seperate_checkbox")
        self.label_11 = QtWidgets.QLabel(self.crop_tab)
        self.label_11.setGeometry(QtCore.QRect(770, 80, 141, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.hist_label_2 = QtWidgets.QLabel(self.crop_tab)
        self.hist_label_2.setGeometry(QtCore.QRect(550, 150, 591, 31))
        self.hist_label_2.setText("")
        self.hist_label_2.setObjectName("hist_label_2")
        self.widget = QtWidgets.QWidget(self.crop_tab)
        self.widget.setGeometry(QtCore.QRect(539, 170, 600, 511))
        self.widget.setObjectName("widget")
        self.hist_label = QtWidgets.QLabel(self.crop_tab)
        self.hist_label.setGeometry(QtCore.QRect(550, 130, 591, 31))
        self.hist_label.setText("")
        self.hist_label.setObjectName("hist_label")
        self.seperate_oclass = QtWidgets.QCheckBox(self.crop_tab)
        self.seperate_oclass.setGeometry(QtCore.QRect(310, 20, 141, 18))
        self.seperate_oclass.setObjectName("seperate_oclass")
        self.tabWidget.addTab(self.crop_tab, "")
        self.image_extractor_tab = QtWidgets.QWidget()
        self.image_extractor_tab.setObjectName("image_extractor_tab")
        self.label_5 = QtWidgets.QLabel(self.image_extractor_tab)
        self.label_5.setGeometry(QtCore.QRect(150, 10, 600, 600))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.tabWidget.addTab(self.image_extractor_tab, "")
        self.duplicate_tab = QtWidgets.QWidget()
        self.duplicate_tab.setObjectName("duplicate_tab")
        self.seek_identical = QtWidgets.QPushButton(self.duplicate_tab)
        self.seek_identical.setGeometry(QtCore.QRect(20, 30, 171, 31))
        self.seek_identical.setObjectName("seek_identical")
        self.tableWidget = QtWidgets.QTableWidget(self.duplicate_tab)
        self.tableWidget.setGeometry(QtCore.QRect(20, 70, 1181, 641))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.horizontalSlider = QtWidgets.QSlider(self.duplicate_tab)
        self.horizontalSlider.setGeometry(QtCore.QRect(800, 110, 191, 22))
        self.horizontalSlider.setSingleStep(1)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.hist_analysis = QtWidgets.QCheckBox(self.duplicate_tab)
        self.hist_analysis.setGeometry(QtCore.QRect(800, 20, 141, 18))
        self.hist_analysis.setObjectName("hist_analysis")
        self.hist_min = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_min.setGeometry(QtCore.QRect(800, 80, 51, 16))
        self.hist_min.setObjectName("hist_min")
        self.hist_max = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_max.setGeometry(QtCore.QRect(940, 80, 51, 16))
        self.hist_max.setObjectName("hist_max")
        self.hist_value = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_value.setGeometry(QtCore.QRect(1010, 110, 51, 16))
        self.hist_value.setObjectName("hist_value")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.duplicate_tab)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(800, 140, 191, 22))
        self.horizontalSlider_2.setProperty("value", 99)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.hist_lower = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_lower.setGeometry(QtCore.QRect(740, 110, 41, 20))
        self.hist_lower.setObjectName("hist_lower")
        self.hist_upper = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_upper.setGeometry(QtCore.QRect(740, 140, 41, 20))
        self.hist_upper.setObjectName("hist_upper")
        self.hist_value_2 = QtWidgets.QLabel(self.duplicate_tab)
        self.hist_value_2.setGeometry(QtCore.QRect(1010, 140, 51, 16))
        self.hist_value_2.setObjectName("hist_value_2")
        self.label_7 = QtWidgets.QLabel(self.duplicate_tab)
        self.label_7.setGeometry(QtCore.QRect(720, 220, 501, 431))
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.similar_analysis = QtWidgets.QCheckBox(self.duplicate_tab)
        self.similar_analysis.setGeometry(QtCore.QRect(800, 50, 131, 20))
        self.similar_analysis.setObjectName("similar_analysis")
        self.similar_groups = QtWidgets.QComboBox(self.duplicate_tab)
        self.similar_groups.setGeometry(QtCore.QRect(740, 80, 411, 21))
        self.similar_groups.setObjectName("similar_groups")
        self.label_8 = QtWidgets.QLabel(self.duplicate_tab)
        self.label_8.setGeometry(QtCore.QRect(760, 250, 400, 400))
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.duplicate_tab)
        self.label_9.setGeometry(QtCore.QRect(950, 20, 58, 16))
        self.label_9.setObjectName("label_9")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.duplicate_tab)
        self.doubleSpinBox.setGeometry(QtCore.QRect(950, 40, 62, 22))
        self.doubleSpinBox.setDecimals(2)
        self.doubleSpinBox.setMinimum(0.01)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.5)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.spinBox_ROI = QtWidgets.QSpinBox(self.duplicate_tab)
        self.spinBox_ROI.setGeometry(QtCore.QRect(1030, 40, 51, 22))
        self.spinBox_ROI.setMinimum(50)
        self.spinBox_ROI.setMaximum(350)
        self.spinBox_ROI.setSingleStep(10)
        self.spinBox_ROI.setObjectName("spinBox_ROI")
        self.label_12 = QtWidgets.QLabel(self.duplicate_tab)
        self.label_12.setGeometry(QtCore.QRect(1040, 20, 19, 16))
        self.label_12.setObjectName("label_12")
        self.delete_group = QtWidgets.QPushButton(self.duplicate_tab)
        self.delete_group.setGeometry(QtCore.QRect(1120, 120, 75, 21))
        self.delete_group.setObjectName("delete_group")
        self.Save_uniqe = QtWidgets.QPushButton(self.duplicate_tab)
        self.Save_uniqe.setEnabled(True)
        self.Save_uniqe.setGeometry(QtCore.QRect(1120, 150, 75, 21))
        self.Save_uniqe.setObjectName("Save_uniqe")
        self.No_features = QtWidgets.QCheckBox(self.duplicate_tab)
        self.No_features.setGeometry(QtCore.QRect(1090, 20, 121, 20))
        self.No_features.setObjectName("No_features")
        self.tabWidget.addTab(self.duplicate_tab, "")
        self.Source_TextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.Source_TextEdit.setGeometry(QtCore.QRect(10, 40, 221, 31))
        self.Source_TextEdit.setPlainText("")
        self.Source_TextEdit.setObjectName("Source_TextEdit")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 150, 47, 14))
        self.label_10.setObjectName("label_10")
        self.source_button = QtWidgets.QPushButton(self.centralwidget)
        self.source_button.setGeometry(QtCore.QRect(10, 10, 141, 23))
        self.source_button.setObjectName("source_button")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(10, 170, 201, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 200, 101, 21))
        self.label_4.setObjectName("label_4")
        self.Log_listwidget = QtWidgets.QListWidget(self.centralwidget)
        self.Log_listwidget.setGeometry(QtCore.QRect(10, 220, 211, 581))
        self.Log_listwidget.setObjectName("Log_listwidget")
        self.destination_TextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.destination_TextEdit.setGeometry(QtCore.QRect(10, 110, 221, 31))
        self.destination_TextEdit.setPlainText("")
        self.destination_TextEdit.setObjectName("destination_TextEdit")
        self.destination_button = QtWidgets.QPushButton(self.centralwidget)
        self.destination_button.setGeometry(QtCore.QRect(10, 80, 171, 23))
        self.destination_button.setObjectName("destination_button")
        self.Extractor_next = QtWidgets.QPushButton(self.centralwidget)
        self.Extractor_next.setGeometry(QtCore.QRect(790, 780, 75, 23))
        self.Extractor_next.setObjectName("Extractor_next")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(406, 750, 321, 20))
        self.label_6.setObjectName("label_6")
        self.Extractor_prev = QtWidgets.QPushButton(self.centralwidget)
        self.Extractor_prev.setGeometry(QtCore.QRect(710, 780, 75, 23))
        self.Extractor_prev.setObjectName("Extractor_prev")
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(410, 780, 281, 16))
        self.horizontalScrollBar.setMinimum(2)
        self.horizontalScrollBar.setMaximum(10)
        self.horizontalScrollBar.setProperty("value", 10)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1496, 22))
        self.menubar.setObjectName("menubar")
        self.menuImage_Editor = QtWidgets.QMenu(self.menubar)
        self.menuImage_Editor.setObjectName("menuImage_Editor")
        MainWindow.setMenuBar(self.menubar)
        self.menuImage_Editor.addSeparator()
        self.menubar.addAction(self.menuImage_Editor.menuAction())
        self.curr_im = None
        self.Zoom_indx = 15
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Original image size"))
        self.Source_image_size.setText(_translate("MainWindow", "X:              Y:"))
        self.text2.setText(_translate("MainWindow", "X: "))
        self.label_2.setText(_translate("MainWindow", "Output image size"))
        self.text2_2.setText(_translate("MainWindow", "Y:"))
        self.radioButton.setText(_translate("MainWindow", "Add offset"))
        self.y_offset_text.setText(_translate("MainWindow", "Y:"))
        self.x_offset_text.setText(_translate("MainWindow", "X: "))
        self.Star_seperate.setText(_translate("MainWindow", "Start Task"))
        self.Recipe_seperate.setText(_translate("MainWindow", "separate by recipe"))
        self.label_3.setText(_translate("MainWindow", "Number of images per folder"))
        self.Crop_checkbox.setText(_translate("MainWindow", "Crop Images"))
        self.seperate_checkbox.setText(_translate("MainWindow", "Seperate to sub folder"))
        self.label_11.setText(_translate("MainWindow", "Image Histogram"))
        self.seperate_oclass.setText(_translate("MainWindow", "Seperate by  Oclass"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crop_tab), _translate("MainWindow", "Image crop & converter"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.image_extractor_tab), _translate("MainWindow", "Image extractor"))
        self.seek_identical.setText(_translate("MainWindow", "Start searching"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "path 1"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "path 2"))
        self.hist_analysis.setText(_translate("MainWindow", "Enable histogram view"))
        self.hist_min.setText(_translate("MainWindow", "min"))
        self.hist_max.setText(_translate("MainWindow", "max"))
        self.hist_value.setText(_translate("MainWindow", "value"))
        self.hist_lower.setText(_translate("MainWindow", "lower th"))
        self.hist_upper.setText(_translate("MainWindow", "upper th"))
        self.hist_value_2.setText(_translate("MainWindow", "value"))
        self.similar_analysis.setText(_translate("MainWindow", "Find similiar images"))
        self.label_9.setText(_translate("MainWindow", "Thresh hold"))
        self.label_12.setText(_translate("MainWindow", "ROI"))
        self.delete_group.setText(_translate("MainWindow", "delete group"))
        self.No_features.setText(_translate("MainWindow", "No features search"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.duplicate_tab), _translate("MainWindow", "identical image finder"))
        self.label_10.setText(_translate("MainWindow", "Loading"))
        self.source_button.setText(_translate("MainWindow", "Select source directory"))
        self.label_4.setText(_translate("MainWindow", "Online Log view"))
        self.destination_button.setText(_translate("MainWindow", "Select destination directory"))
        self.Extractor_next.setText(_translate("MainWindow", "Next Image"))
        self.label_6.setText(_translate("MainWindow", "0.1                                      Zoom                                    10"))
        self.Extractor_prev.setText(_translate("MainWindow", "Prev Image"))
        self.menuImage_Editor.setTitle(_translate("MainWindow", "Image_Editor"))

class SIFT_Worker(QThread):
    comparison_done = pyqtSignal(QListWidgetItem)  # Signal to indicate the comparison is done
    found_similar_group = pyqtSignal(dict)
    precentage = pyqtSignal(float)
    delta = pyqtSignal(str)
    Images_path_l = pyqtSignal(list)

    def __init__(self, folder_path,th,roi,no_features_flag):
        super(SIFT_Worker, self).__init__()
        self.folder_path = folder_path
        self.ROI = roi
        self.th = th
        self.results = []
        self.no_features_flag = no_features_flag

    def compute_similarity_matrix(self,sift_im_dict):
        image_keys = list(sift_im_dict.keys())
        n = len(image_keys)
        similarity_matrix = np.zeros((n, n))
        bf = cv2.BFMatcher()
        for i in range(n):
            self.Print_log_item(" ".join(["comparing image: ", str(n-i), " out of ",str(n)]), "black")
            for j in range(i + 1, n):
                _, desc1 = sift_im_dict[image_keys[i]]
                _, desc2 = sift_im_dict[image_keys[j]]
                matches = bf.knnMatch(desc1, desc2, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
                similarity_matrix[i, j] = len(good_matches)
                similarity_matrix[j, i] = len(good_matches)
        return similarity_matrix

    def cluster_images(self,similarity_matrix, eps, min_samples,spatial_threshold):
        # Normalize the similarity matrix
        normalized_matrix = similarity_matrix / np.max(similarity_matrix)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        clusters = dbscan.fit_predict(1 - normalized_matrix)

        # Merge clusters based on spatial proximity
        merged_clusters = {}
        for i, cluster_label in enumerate(clusters):
            if cluster_label != -1:  # Ignore noise points
                if cluster_label not in merged_clusters:
                    merged_clusters[cluster_label] = []
                merged_clusters[cluster_label].append(i)

        final_clusters = {}
        visited_clusters = set()

        for cluster_label, cluster_indices in merged_clusters.items():
            if cluster_label not in visited_clusters:
                visited_clusters.add(cluster_label)
                merged_cluster = cluster_indices

                for other_label, other_indices in merged_clusters.items():
                    if other_label not in visited_clusters:
                        if self.are_clusters_close(cluster_indices, other_indices, similarity_matrix,
                                                   spatial_threshold):
                            merged_cluster.extend(other_indices)
                            visited_clusters.add(other_label)

                if len(merged_cluster) >= min_samples:
                    final_clusters[cluster_label] = merged_cluster

        # Assign cluster labels to each image
        final_labels = np.full(len(similarity_matrix), -1)
        for cluster_label, cluster_indices in final_clusters.items():
            for idx in cluster_indices:
                final_labels[idx] = cluster_label

        return final_labels

    def are_clusters_close(self, cluster1, cluster2, similarity_matrix, threshold):
        avg_similarity = 0
        count = 0
        for i in cluster1:
            for j in cluster2:
                avg_similarity += similarity_matrix[i, j]
                count += 1
        if count > 0:
            avg_similarity /= count
        return avg_similarity >= threshold

    def cluster_images2(self,similarity_matrix, eps, min_samples):
        # Normalize the similarity matrix
        normalized_matrix = similarity_matrix / np.max(similarity_matrix)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        clusters = dbscan.fit_predict(1 - normalized_matrix)

        return clusters

    def group_images_by_cluster(self,sift_im_dict, clusters):
        image_keys = list(sift_im_dict.keys())
        cluster_groups = {}
        for i, cluster_label in enumerate(clusters):
            if cluster_label not in cluster_groups:
                cluster_groups[cluster_label] = []
            cluster_groups[cluster_label].append(image_keys[i])
        return cluster_groups


    def run(self):
        groups = []
        correlation_values = []
        global image_dict_path
        similar_images_dict = {}
        #loading images from path to dict
        self.no_featur_list =[]
        self.image_dict = self.load_images_to_dict(self.folder_path)
        self.Images_path_l.emit(list(self.image_dict.keys()))
        self.Print_log_item(" ".join(["start comparing : ", str(len(self.image_dict)), " files"]), "black")
        th = self.th
        Features_images_dict = self.Calc_Sift_Descriptor(self.image_dict)

        similarity_matrix = self.compute_similarity_matrix(Features_images_dict)

        # Perform DBSCAN clustering
        eps = 0.2 # Adjust the epsilon value based on your data
        min_samples = 2  # Adjust the minimum number of samples required to form a cluster
        spatial_threshold = 0.90
        clusters = self.cluster_images(similarity_matrix, eps, min_samples,spatial_threshold)

        # Group images by cluster labels
        cluster_groups = self.group_images_by_cluster(Features_images_dict, clusters)
        indx = 0
        str_tmp = 'group_'
        new_dict = {}
        for cluster_label, c_list in cluster_groups.items():
            if cluster_label != -1:
                new_dict[str_tmp + str(indx)] = [c_list]
                indx += 1

        if -1 in cluster_groups:
            new_dict['unclustered'] = cluster_groups[-1]


        time.sleep(0.001)
        similar_images_dict["no_feature"] = self.no_featur_list
        self.found_similar_group.emit(new_dict)
        self.Print_log_item(" ".join(["Comparison done, found: ", str(len(similar_images_dict)), " groups of similar images"]), "green")

    def Calc_Sift_Descriptor(self,im_dict):
        Sift_im_dict={}
        for image in im_dict.keys():
            croped_image_ref = self.resize_image_arr(im_dict[image])
            akaze_params = {
                'descriptor_type': cv2.AKAZE_DESCRIPTOR_MLDB,  # You can adjust this based on your needs
                'descriptor_size': 0,  # The default value (0) uses the optimal size for the descriptor type
                'descriptor_channels': croped_image_ref.shape[-1],
                'threshold': 0.0001,  # Adjust this threshold to control the number of keypoints detected
                'nOctaves': 4,  # Number of octaves in the scale-space pyramid
                'nOctaveLayers': 4,  # Number of layers per octave
                'diffusivity': cv2.KAZE_DIFF_PM_G1  # Choose the diffusivity type
            }
            akaze = cv2.AKAZE_create(**akaze_params)
            keypoints, descriptors = akaze.detectAndCompute(croped_image_ref, None)
            norms = np.linalg.norm(descriptors, axis=1)

            # Sort the descriptors based on their norms in descending order
            sorted_indices = np.argsort(norms)[::-1]

            # Select the top-k most important descriptors
            k = 50  # Specify the number of important descriptors to select
            important_descriptors = descriptors[sorted_indices[:k]]
            Sift_im_dict[image] = [keypoints, important_descriptors]
        return Sift_im_dict

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
            'threshold': 0.0001,  # Adjust this threshold to control the number of keypoints detected
            'nOctaves': 4,  # Number of octaves in the scale-space pyramid
            'nOctaveLayers': 4,  # Number of layers per octave
            'diffusivity': cv2.KAZE_DIFF_PM_G1  # Choose the diffusivity type
        }
        akaze = cv2.AKAZE_create(**akaze_params)
        th = self.th
        try:
            croped_image_ref = self.resize_image_arr(image)
            keypoints_ref, descriptors_ref = akaze.detectAndCompute(croped_image_ref, None)
            #image_with_keypoints = cv2.drawKeypoints(croped_image_ref, keypoints_ref, None)

            # Display the image with keypoints

            if descriptors_ref is None or descriptors_ref.shape[0] == 0:
                self.Print_log_item("empty Descriptors for: " + self.Ref_image_path,"red")
                self.no_featur_list.append(self.Ref_image_path)
                return None

            if self.no_features_flag == False:
                for img in img_dict.keys():
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
                    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    #matches = bf.match(descriptors_ref, descriptors_target)
                    # Apply ratio test to filter good matches
                    # Create BF Matcher
                    bf_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)

                    # Convert descriptors to GPU mat
                    des1_gpu = cv2.cuda_GpuMat()
                    des2_gpu = cv2.cuda_GpuMat()
                    des1_gpu.upload(descriptors_ref)
                    des2_gpu.upload(descriptors_target)

                    # Perform matching
                    matches = bf_matcher.match(des1_gpu, des2_gpu)

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

            time.sleep(0.001)
            res = [similar_group,correlation_values]
            return (res)
        except FileNotFoundError as e:
            self.Print_log_item(["An error occurred:", str(e)], "red")

    def Print_log_item(self,str,color):
        item = QListWidgetItem(str)
        item.setForeground(QColor(color))  # Set the font color to red
        self.comparison_done.emit(item)
        time.sleep(0.001)
        #return item

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

#------------- initialize all clickes --------------------------------------------------------------------------------------------------------------
    def init_button_actions(self):
        self.radioButton.clicked.connect(self.checkBox_set_offset)
        self.delete_group.clicked.connect(self.delete_similar_group)
        self.Save_uniqe.clicked.connect(self.Save_uniqe_im)
        self.source_button.clicked.connect(self.select_folder)
        self.destination_button.clicked.connect(self.select_folder)
        self.Star_seperate.clicked.connect(self.seperate_images)
        self.seek_identical.clicked.connect(self.start_analysis)
        self.tableWidget.clicked.connect(self.table_clicked)
        self.Extractor_next.clicked.connect(self.toggle_image_in_list)
        self.Extractor_prev.clicked.connect(self.toggle_image_in_list)
        self.hist_analysis.clicked.connect(self.checkbox_histogram_view)
        self.horizontalSlider.valueChanged.connect(self.Scrollbar_lower_th_display)
        self.horizontalSlider_2.valueChanged.connect(self.Scrollbar_upper_th_display)
        self.horizontalScrollBar.valueChanged.connect(self.Event_scrollbar)
        self.similar_analysis.clicked.connect(self.checkbox_similar_images)
        self.similar_groups.activated.connect(self.display_comboBox_group)
        self.Crop_checkbox.clicked.connect(self.checkbox_manager)
        self.seperate_oclass.clicked.connect(self.checkbox_manager)
        self.seperate_checkbox.clicked.connect(self.checkbox_manager)
        self.spinBox_ROI.valueChanged.connect(self.RoI_update)
        self.init_appereance()

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
        self.delete_group.hide()
        self.Save_uniqe.hide()
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
        self.press_mouse_pos = None
        self.line_hist_flag = 0

#-----------------------------------------checkbox-----------------------------------------------------
    def checkbox_manager(self):
        sender = self.sender().objectName()
        if sender == "Crop_checkbox":
            self.seperate_checkbox.setChecked(0)
            self.seperate_oclass.setChecked(0)
            self.plainTextEdit.hide()
            self.Recipe_seperate.hide()
            self.label_2.show()
            self.radioButton.show()
            self.text2.show()
            self.output_X_size.show()
            self.output_Y_size.show()
            self.text2_2.show()
        elif sender == "seperate_oclass":

            self.Crop_checkbox.setChecked(0)
            self.seperate_checkbox.setChecked(0)
            self.plainTextEdit.hide()
            self.Recipe_seperate.hide()
            self.label_2.hide()
            self.radioButton.hide()
            self.text2.hide()
            self.output_X_size.hide()
            self.output_Y_size.hide()
            self.text2_2.hide()
            self.checkBox_set_offset()
            self.radioButton.setChecked(0)
        elif sender == "seperate_checkbox":
            self.Crop_checkbox.setChecked(0)
            self.seperate_oclass.setChecked(0)
            self.plainTextEdit.show()
            self.Recipe_seperate.show()
            self.label_2.hide()
            self.radioButton.hide()
            self.text2.hide()
            self.output_X_size.hide()
            self.output_Y_size.hide()
            self.text2_2.show()
            self.checkBox_set_offset()
            self.radioButton.setChecked(0)
        elif sender == "radioButton":
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
            self.Save_uniqe.hide()

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
        if (self.radioButton.isChecked() and  self.Crop_checkbox.isChecked()) :
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

    def Checkbox_seperate_oclass(self):
        self.Crop_seperate_toggle(self.seperate_oclass.isChecked())

#---------------------------------------------Push buttons-------------------------------------------------------------
    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="select folder")
        sender = self.sender().objectName()


        if sender == "source_button":
            self.Source_path = folder_path
            self.root_folder = folder_path
            self.Source_TextEdit.setPlainText(folder_path)
            self.files_list = self.get_image_list_from_root(self.Source_path)
            self.f_object = [self.files_list, 0]
            self.current_image_path = self.find_first_image(self.Source_path)
            self.curr_im = Image.open(self.current_image_path)
            self.Update_image()

        else:
            self.destination_TextEdit.setPlainText(folder_path)
            self.destination_path = folder_path

    def toggle_image_in_list(self):
        list = self.f_object[0]
        l = len(list)
        try:
            sender = self.sender().objectName()
        except:
            sender = "Extractor_next"

        if sender == "Extractor_next":
            tmp_str = "next"
            self.Extractor_prev.show()
            file_indx = self.f_object[1] + 1
            if l - 1 == file_indx:
                self.Extractor_next.hide()
        else:
            tmp_str = "prev"
            self.Extractor_next.show()
            file_indx = self.f_object[1] - 1
            if file_indx == 0:
                self.Extractor_prev.hide()
        self.f_object[1] = file_indx
        self.current_image_path = list[file_indx]
        self.temp_img = self.current_image_path
        self.write_to_logview( tmp_str + "image " + str(file_indx) + "/" + str(len(list)))
        self.Update_image()

    def Update_image(self):
        try:
            self.image_changing(self.current_image_path)

            im = Image.open(self.current_image_path)
            self.temp_img = self.current_image_path
            self.image_hist_plot(im)
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            if self.tabWidget.currentWidget().objectName() == 'crop_tab':
                geo = self.label_image.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_image.setPixmap(pixmap)
            elif self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
                geo = self.label_5.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_5.setPixmap(pixmap)
            else:
                geo = self.label_8.geometry().getRect()
                pixmap = pixmap.scaled(geo[-1], geo[-1])
                self.label_8.setPixmap(pixmap)
        except:
            QMessageBox.about(self, "info massage", "no more images")

    # show next image in list
    def delete_similar_group(self):
        xx= self.similar_groups.currentText()
        for img in list(self.candidates[xx]['Images']):
            os.remove(img)

    def Save_uniqe_im(self):
        x=1
        Exclude_img_list=[]
        for img_list in list(self.candidates.values()):
            if  len(img_list)>0:
                for img in list(img_list['Images']):
                    Exclude_img_list.append(img)
        for img in self.img_path_list:
            if not(img in Exclude_img_list):
                image = Image.open(img)
                img_name=img.split("\\")[-1]
                image.save(os.path.join(self.destination_path,img_name))

    def delete_similar_groups(self):
        for group in self.candidates.keys():
            for image in self.candidates[group]["Images"]:
                os.remove(image)

# ------------------------------ GUI Updates---------------
    def image_hist_plot(self,im):
        bands = len(im.getbands())
        arr = np.array(im,dtype=object)
        hist_list = []
        if bands == 1:
            hist_list.append(np.histogram(arr, bins=256, range=(0, 255)))
        else:
            hist_list.append(np.histogram(arr[:,:,0], bins=256, range=(0, 256)))
            hist_list.append(np.histogram(arr[:,:,1], bins=256, range=(0, 256)))
            hist_list.append(np.histogram(arr[:,:,2], bins=256, range=(0, 256)))
        self.plot_widget.clear()
        self.pygraph_plot(hist_list)

    def pygraph_plot(self, hist_list):
        self.plot_widget.clear()
        colors = ["red", "green", "blue"]
        peaks_string =""
        RGB_string = ""
        if len(hist_list) == 1:
            colors = ["white"]
        fill_color = [(255,0,0,100),(0,255,0,100),(0,0,255,100)]
        for i,channel_hist in enumerate(hist_list):
            counts, bins = channel_hist
            max_indices = np.argpartition(channel_hist[0], -2)[-2:]
            if self.line_hist_flag == 1:
                hist = counts
            else:
                hist = gaussian_filter1d(counts, sigma=5)
            peaks, _ = find_peaks(hist, height=100)
            RGB_string = RGB_string + colors[i] + ": Max peak=" + str(max_indices[0]) + " ,"
            peaks_string = peaks_string + colors[i] + " Peaks:"
            for peak in peaks:
                peaks_string = peaks_string + " " + str(peak) +", "

            bins_adjusted = bins[:]
            hist = hist.astype(np.float32)
            bins_adjusted = bins_adjusted.astype(np.float32)
            self.plot_widget.plot(bins_adjusted[1:], hist[1:], stepMode=True, fillLevel=None, brush=fill_color[i],
                                    pen=colors[i], linewidth=10)



            infinite_line = pg.InfiniteLine(pos=max_indices[0], angle=90)
            pen = pg.mkPen(color=colors[i], style=QtCore.Qt.DashLine)  # Set pen style to DashLine
            infinite_line.setPen(pen)
            self.plot_widget.addItem(infinite_line)

        self.hist_label.setText(RGB_string)
        self.hist_label_2.setText(peaks_string)

    def display_comboBox_group(self):
        flag = 0
        group = self.similar_groups.currentText()
        if group == "no_feature":
            group_len = len(self.candidates[group])
            if group_len > 0:
                img_list =  list(self.candidates[group])
                img_str = self.root_folder + "\\" +img_list[0].split("\\")[-1]
        else:
            img_list =  list(self.candidates[group])[0]
            img_str = self.similar_groups.currentText()
        if 'img_list' in locals():
            self.Add_items_to_table(img_list,flag)
            self.curr_im = Image.open(os.path.normpath(img_list[0]))
            pixmap = self.draw_ROI()
            self.label_8.setPixmap(pixmap)
            self.sub_window.duplicates2 = img_list
            self.delete_group.show()
            self.Save_uniqe.show()

    def update_time_in_log(self, t):
        self.write_to_logview("Compare took: " + str(t) + " seconds")

    def update_im_list(self,l):
        self.img_path_list=l

    def update_bar(self, p):
        self.progressBar.setValue(int(p))

    def update_similar_group(self, similar_dict):
        self.similar_groups.clear()
        similar_groups = list(similar_dict.keys())
        self.candidates = similar_dict
        self.similar_groups.addItems(similar_groups)

    def update_list_items(self, results):
        # Update the ListView with the list of results
        self.write_to_logview(results)

    def image_changing(self, path):
        zoom = self.Zoom_indx
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
        BBOX = middle_frame/self.horizontalScrollBar.value()
        draw.rectangle(
            [middle_frame - BBOX, middle_frame - BBOX, middle_frame + BBOX,
             middle_frame + BBOX],
            outline="red", width=2)
        i.save(os.getcwd() + '\\tmp.jpeg')

    def Crop_seperate_toggle(self,crop):
        sender = self.sender().objectName()
        if sender == 'seperate_checkbox':
            if not crop:
                case = 1
            else:
                case = 0
        elif sender == 'seperate_oclass':
            case = 2
        else:
            if not crop:
                case = 1
            else:
                case = 0
        self.hide_show_prop(case)

    def hide_show_prop(self,case):
        if case == 1:
            self.label_3.hide()
            self.seperate_checkbox.setChecked(1)
            self.Crop_checkbox.setChecked(0)
            self.seperate_oclass.setChecked(0)
            self.plainTextEdit.hide()
            self.Recipe_seperate.hide()
            self.label_2.show()
            self.radioButton.show()
            self.text2.show()
            self.output_X_size.show()
            self.output_Y_size.show()
            self.text2_2.show()

        elif case == 2:
            self.label_3.hide()
            self.seperate_checkbox.setChecked(0)
            self.Crop_checkbox.setChecked(0)
            self.plainTextEdit.hide()
            self.Recipe_seperate.hide()
            self.label_2.hide()
            self.radioButton.hide()
            self.text2.hide()
            self.output_X_size.hide()
            self.output_Y_size.hide()
            self.text2_2.hide()

        else:
            self.seperate_oclass.setChecked(0)
            self.label_3.show()
            self.seperate_checkbox.setChecked(0)
            self.Crop_checkbox.setChecked(1)
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

        scrollDistance = event.angleDelta().y()
        numDegrees = event.angleDelta() / 8
        numSteps = numDegrees / 15
        if scrollDistance > 0:
            zoom_indx = -1*numSteps.manhattanLength()
        else:
            zoom_indx = numSteps.manhattanLength()
        s_indx = self.Zoom_indx
        self.Zoom_indx = s_indx + zoom_indx

        list = self.f_object[0]
        file_indx = self.f_object[1]
        self.curr_im = list[file_indx]
        self.image_changing(self.curr_im)

        pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')

        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            geo = self.label_5.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_5.setPixmap(pixmap)
        else:
            geo = self.label_image.geometry().getRect()
            pixmap = pixmap.scaled(geo[-1], geo[-1])
            self.label_image.setPixmap(pixmap)

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
                self.curr_im = list[file_indx]
                self.image_changing(self.curr_im)

                pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')


                if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
                    geo = self.label_5.geometry().getRect()
                    pixmap = pixmap.scaled(geo[-1], geo[-1])
                    self.label_5.setPixmap(pixmap)
                else:
                    geo = self.label_image.geometry().getRect()
                    pixmap = pixmap.scaled(geo[-1], geo[-1])
                    self.label_image.setPixmap(pixmap)

    # mouse clicked on image to cropp by center
    def mousePressEvent(self, event):
        if self.tabWidget.currentWidget().objectName() == 'image_extractor_tab':
            x = event.x()  - 412
            y = event.y() - 53
            if not (x < 0 or y < 0 or x > 600 or y > 600):
                if self.destination_TextEdit.toPlainText() == "":
                    messagebox.showinfo(title='Error massage', message='please select destination folder')
                else:
                    img2 =  self.center_and_crop_image(self.label_5.geometry().getRect(),self.current_image_path,x,y)
                    if not img2 == None:
                        tmp_str = self.current_image_path.split('\\')
                        img_name = tmp_str[-1]
                        while os.path.exists(self.destination_path + '\\_' + img_name) or os.path.exists(self.destination_path + '\\' + img_name):
                            tmp = img_name.split(".jpeg")
                            img_name = tmp[0] + '_' + '.jpeg'
                        img2.save(self.destination_path + '\\_' + img_name)
                        self.write_to_logview(img_name + ' image was saved')
                        self.toggle_image_in_list()
        elif self.tabWidget.currentWidget().objectName() == 'crop_tab':
            if self.press_mouse_pos == None:
                x = event.x()  - 260
                y = event.y() - 220
                if not (x < 0 or y < 0 or x > 500 or y > 500):
                    geo = self.label_image.geometry()
                    self.press_mouse_pos = (x,y)

    def mouseReleaseEvent(self, event):
        self.plot_line_hist(event)


    def plot_line_hist(self,event):
        if not self.press_mouse_pos == None:
            hist_list = []
            x = event.x() - 262
            y = event.y() - 220
            if not (x < 0 or y < 0 or x > 500 or y > 500):
                image = Image.open(self.temp_img)
                im_dim = image.size
                geo = self.label_image.geometry().getRect()
                factor_x = geo[2] / im_dim[0]
                factor_y = geo[3] / im_dim[1]
                im_arr = cv2.imread(self.temp_img)
                dim = im_arr.shape[2]
                for i in range(dim):
                    tmp_arr = im_arr[:, :, i]
                    line_pixels = self.extract_line_pixels(tmp_arr, (
                    int(self.press_mouse_pos[0] / factor_x), int(self.press_mouse_pos[1] / factor_y), int(x / factor_x),
                    int(y / factor_y)))
                    line_pixels = numpy.array(line_pixels)
                    hist_list.append((line_pixels, np.arange(0, len(line_pixels) + 1)))
                self.line_hist_flag = 1
                hist_list = numpy.asarray(hist_list, dtype=object)
                self.pygraph_plot(hist_list)
                self.draw_line((self.press_mouse_pos[0], self.press_mouse_pos[1]), (x, y))
            self.press_mouse_pos = None

    def draw_line(self,start,end):
        image = Image.open(self.temp_img)
        dim = image.size
        geo = self.label_image.geometry().getRect()
        factor_x = geo[2]/dim[0]
        factor_y = geo[3]/dim[1]
        draw = ImageDraw.Draw(image)

        # Draw a red line on the image
        line_color = "gray"

        line_width = 4
        draw.line([(start[0]/factor_x,start[1]/factor_y),(end[0]/factor_x,end[1]/factor_y)], fill=line_color, width=line_width)
        image.save(os.getcwd() + '\\tmp.jpeg')
        pixmap=QPixmap(os.getcwd() + '\\tmp.jpeg')

        pixmap = pixmap.scaled(geo[-1], geo[-1])
        self.label_image.setPixmap(pixmap)

    def extract_line_pixels(self,image,pos):
        line_points = self.bresenham_line(pos[0], pos[1], pos[2], pos[3])
        extracted_values=[]
        for point in line_points:
            extracted_values.append(image[point[1], point[0]])
        return extracted_values

    def bresenham_line(self,x0, y0, x1, y1):
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1

        swapped = False
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            swapped = True

        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx // 2
        ystep = 1 if y0 < y1 else -1
        y = y0

        points = []
        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            points.append(coord)
            error -= dy
            if error < 0:
                y += ystep
                error += dx

        if swapped:
            points.reverse()
        return points
    # open Image in plot view from table
    def table_clicked(self):
        self.sub_window.ROI = self.spinBox_ROI.value()
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
        no_features_flag = self.No_features.isChecked()
        worker = SIFT_Worker(folder_path,self.doubleSpinBox.value(),self.spinBox_ROI.value(),no_features_flag)
        worker.comparison_done.connect(self.update_list_items)
        worker.found_similar_group.connect(self.update_similar_group)
        worker.precentage.connect(self.update_bar)
        worker.delta.connect(self.update_time_in_log)
        worker.Images_path_l.connect(self.update_im_list)
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
        params = {}
        if not self.Source_path == "":
            if not self.destination_path =="":
                #Simple seperation without recipe
                if self.Crop_checkbox.isChecked():
                    if not (self.output_X_size.toPlainText() == "" and self.output_Y_size.toPlainText()  == ""):
                        params["output x"] = int(self.output_X_size.toPlainText())
                        params["output y"] = int(self.output_Y_size.toPlainText())
                        if self.radioButton.isChecked():
                            params["offset x"] = int(self.offset_x.toPlainText())
                            params["offset y"] = int(self.offset_y.toPlainText())
                        else:
                            params["offset x"] = 0
                            params["offset y"] = 0
                        worker = Image_processing_Worker(self.Source_path,self.destination_path, "crop", params)

                    else:
                        messagebox.showinfo(title='Error massage', message='Please fill in the output size')
                if self.seperate_checkbox.isChecked():
                    if not (self.offset_x.toPlainText()=="" and self.offset_y.toPlainText()==""):
                    # Seperate folders
                        params = [int(self.plainTextEdit.toPlainText())]
                        worker = Image_processing_Worker(self.Source_path,self.destination_path, "folders", params)
                    else:
                            messagebox.showinfo(title='Error massage', message='Please fill in the offset')
                    self.write_to_logview("finish convert")
                if self.seperate_oclass.isChecked():
                    params = [None]
                    worker = Image_processing_Worker(self.Source_path, self.destination_path, "folders", params)
                worker.log_write.connect(self.update_list_items)
                worker.start()
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

    def center_and_crop_image(self,img_rect,cuur_path,x,y):

        img_dim = Image.open(cuur_path).size
        if (not(x < 0 or y < 0 or x > 600 or y > 600) and (not self.seperate_oclass.isChecked())):

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
                cropped_image = img1.crop((0, 0, img_dim[0], img_dim[1]))
        elif self.seperate_oclass.isChecked():
            img = Image.open(cuur_path)
            img1 = Image.new(img.mode, (img_dim[0], img_dim[1]) )
            offset_x = int(img_dim[0] // 2 -x)
            offset_y = int(img_dim[1] // 2 -y)
            img1.paste(img, (0 + offset_x, 0 + offset_y))

            # Crop the image
            cropped_image = img1.crop((0, 0, img_dim[0], img_dim[1]))
        return cropped_image

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
        if len(duplicate_paths)>0:
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

    def RoI_update(self):
        p=self.draw_ROI()
        self.label_8.setPixmap(p)

    def draw_ROI(self):
        #im = self.curr_im
        ROI = self.spinBox_ROI.value()
        if not (self.curr_im is None):
            geo = self.label_8.geometry().getRect()
            resized_im = self.curr_im.resize([geo[2], geo[3]])
            draw = ImageDraw.Draw(resized_im)
            middle_frame = geo[2] / 2
            draw.rectangle([middle_frame - ROI / 2,
                            middle_frame - ROI / 2,
                            middle_frame + ROI / 2,
                            middle_frame + ROI / 2],
                           outline="blue", width=2)
            resized_im.save(os.getcwd() + '\\tmp.jpeg')
            pixmap = QPixmap(os.getcwd() + '\\tmp.jpeg')
            return(pixmap)

class Image_processing_Worker(QThread):
    log_write = pyqtSignal(str)

    def __init__(self, source_path,destination_path,state,params):
        super(Image_processing_Worker, self).__init__()
        self.source_path = source_path
        self.destination_path = destination_path
        self.params = params
        self.state = state

    def Print_log_item(self,str,color):
        item = QListWidgetItem(str)
        item.setForeground(QColor(color))  # Set the font color to red
        self.log_write.emit(item)

    def run(self):
        if self.state == "crop":
            self.crop_images()
        elif self.state == "seperate by folders":
            self.seperate_by_folders()
        else:
            self.seperate_oclass()

    def seperate_oclass(self):
        for root, dirs, files in os.walk(self.source_path):
            if "ADC" in dirs:
                df = pd.read_csv(root + '\\ADC\\Surface2Bump.csv')
                l = len(df)
                for i in range(l):
                    image_name = df["ImageName"][i]
                    image_bin = df[" OClass"][i]
                    image_x = df["ImageX"][i]
                    image_y = df["ImageY"][i]
                    if not os.path.exists(os.path.normpath(self.destination_path + '\\' + str(image_bin))):
                        os.makedirs(os.path.normpath(self.destination_path + '\\' + str(image_bin)))

                    src = (root + '\\' + image_name)
                    tmp_img = self.center_and_crop_image((1000, 1000), src, image_x, image_y)
                    dst = os.path.normpath(self.destination_path + '\\' + str(image_bin) + "\\" + image_name)
                    tmp_img.save(dst)

    def crop_images(self):
        # Crop and offset
        self.log_write.emit("Start cropping images")

        indx = 1
        new_width = self.params["output x"]
        new_height = self.params["output y"]
        SourcePath_len = len(self.source_path)

        for root, dirs, files in os.walk(self.source_path):
            for file_ in files:
                tmp_str = file_.split('.')
                if tmp_str[-1].lower() in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']:
                    img = Image.open(root + "\\" + file_)
                    w, h = img.size
                    destination_Path = os.path.normpath(self.destination_path + root[SourcePath_len:])
                    is_dir = os.path.isdir(destination_Path)
                    if is_dir == False:
                        os.makedirs(destination_Path)

                    img1 = self.Image_convert(math.ceil((w - new_width) / 2),
                                                  math.ceil((h - new_height) / 2), img)

                    img1.save(destination_Path + '\\' + file_)
                    self.log_write.emit("Image: " + str(indx) + " " + file_ + " converted")
                indx = indx + 1

        self.log_write.emit("finished cropping images")

    def seperate_by_folders(self):
        N = int(self.plainTextEdit.toPlainText())  # number of images per subfolder
        self.write_to_logview("start seperation into subfolders - each folder contains: " + str(N))
        folder_num = 0
        # create the first folder0
        folder_path = os.path.join(self.destination_path, "folder" + str(folder_num))
        os.makedirs(folder_path)
        # loop throught all subfolders in path
        for root, dirs, files in os.walk(self.source_path):
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

    def Image_convert(self ,left ,top ,img):
        new_width = self.params["output x"]
        new_height = self.params["output y"]
        w,h = img.size
        if self.params["offset x"] != 0 or self.params["offset x"] != 0:
            img1 = Image.new(img.mode, (w + self.params["offset x"], h + self.params["offset y"]))
            img1.paste(img, ( - self.params["offset x"], - self.params["offset y"]))
        else:
            img1 =img
        left = math.ceil((w - new_width) / 2)
        top = math.ceil((h - new_height) / 2)
        right = math.ceil((w + new_width) / 2)
        bottom = math.ceil((h + new_height) / 2)
        img2 = img1.crop((left, top, right, bottom))
        return img2

if __name__ == "__main__":
    app=QApplication(sys.argv)
    main_win=UI()
    main_win.show()

    app.exec_()