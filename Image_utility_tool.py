# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\doron\Documents\ADC utility tool\Image_utility_tool.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1496, 777)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(260, 0, 1221, 741))
        self.tabWidget.setObjectName("tabWidget")
        self.crop_tab = QtWidgets.QWidget()
        self.crop_tab.setObjectName("crop_tab")
        self.label_image = QtWidgets.QLabel(self.crop_tab)
        self.label_image.setGeometry(QtCore.QRect(20, 210, 400, 400))
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.label = QtWidgets.QLabel(self.crop_tab)
        self.label.setGeometry(QtCore.QRect(30, 620, 101, 21))
        self.label.setObjectName("label")
        self.Source_image_size = QtWidgets.QLabel(self.crop_tab)
        self.Source_image_size.setGeometry(QtCore.QRect(140, 620, 171, 21))
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
        self.tabWidget.addTab(self.crop_tab, "")
        self.image_extractor_tab = QtWidgets.QWidget()
        self.image_extractor_tab.setObjectName("image_extractor_tab")
        self.label_5 = QtWidgets.QLabel(self.image_extractor_tab)
        self.label_5.setGeometry(QtCore.QRect(150, 10, 600, 600))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.Extractor_prev = QtWidgets.QPushButton(self.image_extractor_tab)
        self.Extractor_prev.setGeometry(QtCore.QRect(550, 650, 75, 23))
        self.Extractor_prev.setObjectName("Extractor_prev")
        self.Extractor_next = QtWidgets.QPushButton(self.image_extractor_tab)
        self.Extractor_next.setGeometry(QtCore.QRect(630, 650, 75, 23))
        self.Extractor_next.setObjectName("Extractor_next")
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.image_extractor_tab)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(250, 650, 281, 16))
        self.horizontalScrollBar.setMinimum(1)
        self.horizontalScrollBar.setMaximum(15)
        self.horizontalScrollBar.setProperty("value", 15)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.label_6 = QtWidgets.QLabel(self.image_extractor_tab)
        self.label_6.setGeometry(QtCore.QRect(246, 620, 321, 20))
        self.label_6.setObjectName("label_6")
        self.label_11 = QtWidgets.QLabel(self.image_extractor_tab)
        self.label_11.setGeometry(QtCore.QRect(880, 230, 141, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.hist_label = QtWidgets.QLabel(self.image_extractor_tab)
        self.hist_label.setGeometry(QtCore.QRect(750, 620, 421, 31))
        self.hist_label.setText("")
        self.hist_label.setObjectName("hist_label")
        self.widget = QtWidgets.QWidget(self.image_extractor_tab)
        self.widget.setGeometry(QtCore.QRect(750, 260, 421, 331))
        self.widget.setObjectName("widget")
        self.hist_label_2 = QtWidgets.QLabel(self.image_extractor_tab)
        self.hist_label_2.setGeometry(QtCore.QRect(750, 640, 421, 31))
        self.hist_label_2.setText("")
        self.hist_label_2.setObjectName("hist_label_2")
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
        self.doubleSpinBox.setMinimum(0.1)
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
        self.Log_listwidget.setGeometry(QtCore.QRect(10, 220, 211, 511))
        self.Log_listwidget.setObjectName("Log_listwidget")
        self.destination_TextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.destination_TextEdit.setGeometry(QtCore.QRect(10, 110, 221, 31))
        self.destination_TextEdit.setPlainText("")
        self.destination_TextEdit.setObjectName("destination_TextEdit")
        self.destination_button = QtWidgets.QPushButton(self.centralwidget)
        self.destination_button.setGeometry(QtCore.QRect(10, 80, 171, 23))
        self.destination_button.setObjectName("destination_button")
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

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
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
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.crop_tab), _translate("MainWindow", "Image crop & converter"))
        self.Extractor_prev.setText(_translate("MainWindow", "Prev Image"))
        self.Extractor_next.setText(_translate("MainWindow", "Next Image"))
        self.label_6.setText(_translate("MainWindow", "0.1                                      Zoom                                    10"))
        self.label_11.setText(_translate("MainWindow", "Image Histogram"))
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
        self.menuImage_Editor.setTitle(_translate("MainWindow", "Image_Editor"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
