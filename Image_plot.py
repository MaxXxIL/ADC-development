# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\doron\Documents\ADC utility tool\Image plot.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_plot_image(object):
    def setupUi(self, plot_image):
        plot_image.setObjectName("plot_image")
        plot_image.resize(644, 662)
        self.centralwidget = QtWidgets.QWidget(plot_image)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 10, 501, 501))
        self.label.setText("")
        self.label.setObjectName("label")
        self.Button_prev = QtWidgets.QPushButton(self.centralwidget)
        self.Button_prev.setGeometry(QtCore.QRect(470, 550, 75, 23))
        self.Button_prev.setObjectName("Button_prev")
        self.Button_del = QtWidgets.QPushButton(self.centralwidget)
        self.Button_del.setGeometry(QtCore.QRect(10, 550, 75, 23))
        self.Button_del.setObjectName("Button_del")
        self.Button_next = QtWidgets.QPushButton(self.centralwidget)
        self.Button_next.setGeometry(QtCore.QRect(550, 550, 75, 23))
        self.Button_next.setObjectName("Button_next")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(6, 523, 621, 20))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(90, 550, 361, 21))
        self.label_3.setObjectName("label_3")
        self.Next_row_im = QtWidgets.QPushButton(self.centralwidget)
        self.Next_row_im.setGeometry(QtCore.QRect(310, 590, 111, 23))
        self.Next_row_im.setObjectName("Next_row_im")
        self.Prev_row_im = QtWidgets.QPushButton(self.centralwidget)
        self.Prev_row_im.setGeometry(QtCore.QRect(200, 590, 111, 23))
        self.Prev_row_im.setObjectName("Prev_row_im")
        plot_image.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(plot_image)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 644, 22))
        self.menubar.setObjectName("menubar")
        plot_image.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(plot_image)
        self.statusbar.setObjectName("statusbar")
        plot_image.setStatusBar(self.statusbar)

        self.retranslateUi(plot_image)
        QtCore.QMetaObject.connectSlotsByName(plot_image)

    def retranslateUi(self, plot_image):
        _translate = QtCore.QCoreApplication.translate
        plot_image.setWindowTitle(_translate("plot_image", "MainWindow"))
        self.Button_prev.setText(_translate("plot_image", "Prev image"))
        self.Button_del.setText(_translate("plot_image", "Delete Image"))
        self.Button_next.setText(_translate("plot_image", "Next Image"))
        self.label_2.setText(_translate("plot_image", "Image Path:"))
        self.label_3.setText(_translate("plot_image", "Class Lable:"))
        self.Next_row_im.setText(_translate("plot_image", "Next row image"))
        self.Prev_row_im.setText(_translate("plot_image", "Prev row image"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    plot_image = QtWidgets.QMainWindow()
    ui = Ui_plot_image()
    ui.setupUi(plot_image)
    plot_image.show()
    sys.exit(app.exec_())
