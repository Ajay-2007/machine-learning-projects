# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1095, 638)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1101, 621))
        self.tabWidget.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.tabWidget.setStyleSheet("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tableView = QtWidgets.QTableView(self.tab_3)
        self.tableView.setGeometry(QtCore.QRect(0, 90, 1091, 501))
        self.tableView.setObjectName("tableView")
        self.pushButton = QtWidgets.QPushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(280, 30, 391, 41))
        self.pushButton.setObjectName("pushButton")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_4)
        self.comboBox_3.setGeometry(QtCore.QRect(180, 70, 171, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_4 = QtWidgets.QComboBox(self.tab_4)
        self.comboBox_4.setGeometry(QtCore.QRect(610, 70, 151, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.label_2 = QtWidgets.QLabel(self.tab_4)
        self.label_2.setGeometry(QtCore.QRect(210, 40, 131, 21))
        self.label_2.setObjectName("label_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_5.setGeometry(QtCore.QRect(370, 70, 81, 31))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_6.setGeometry(QtCore.QRect(800, 70, 81, 31))
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_3 = QtWidgets.QLabel(self.tab_4)
        self.label_3.setGeometry(QtCore.QRect(620, 40, 131, 21))
        self.label_3.setObjectName("label_3")
        self.pushButton_7 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_7.setGeometry(QtCore.QRect(480, 130, 121, 31))
        self.pushButton_7.setObjectName("pushButton_7")
        self.tableView_6 = QtWidgets.QTableView(self.tab_4)
        self.tableView_6.setGeometry(QtCore.QRect(270, 180, 271, 401))
        self.tableView_6.setObjectName("tableView_6")
        self.tableView_7 = QtWidgets.QTableView(self.tab_4)
        self.tableView_7.setGeometry(QtCore.QRect(820, 180, 281, 401))
        self.tableView_7.setObjectName("tableView_7")
        self.tableView_8 = QtWidgets.QTableView(self.tab_4)
        self.tableView_8.setGeometry(QtCore.QRect(0, 180, 271, 401))
        self.tableView_8.setObjectName("tableView_8")
        self.tableView_9 = QtWidgets.QTableView(self.tab_4)
        self.tableView_9.setGeometry(QtCore.QRect(540, 180, 281, 401))
        self.tableView_9.setObjectName("tableView_9")
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.checkBox = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox.setGeometry(QtCore.QRect(120, 116, 131, 31))
        self.checkBox.setObjectName("checkBox")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox_2.setGeometry(QtCore.QRect(120, 150, 111, 31))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox_3.setGeometry(QtCore.QRect(120, 190, 91, 21))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox_4.setGeometry(QtCore.QRect(120, 210, 141, 41))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox_5.setGeometry(QtCore.QRect(120, 250, 121, 31))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.tab_5)
        self.checkBox_6.setGeometry(QtCore.QRect(120, 280, 91, 31))
        self.checkBox_6.setObjectName("checkBox_6")
        self.label_4 = QtWidgets.QLabel(self.tab_5)
        self.label_4.setGeometry(QtCore.QRect(110, 40, 241, 31))
        self.label_4.setObjectName("label_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_3.setGeometry(QtCore.QRect(510, 420, 101, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_5 = QtWidgets.QLabel(self.tab_5)
        self.label_5.setGeometry(QtCore.QRect(390, 40, 211, 31))
        self.label_5.setObjectName("label_5")
        self.label = QtWidgets.QLabel(self.tab_5)
        self.label.setGeometry(QtCore.QRect(410, 120, 51, 21))
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_7 = QtWidgets.QLabel(self.tab_5)
        self.label_7.setGeometry(QtCore.QRect(410, 150, 51, 21))
        self.label_7.setText("")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab_5)
        self.label_8.setGeometry(QtCore.QRect(410, 180, 51, 21))
        self.label_8.setText("")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab_5)
        self.label_9.setGeometry(QtCore.QRect(410, 210, 51, 21))
        self.label_9.setText("")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.tab_5)
        self.label_10.setGeometry(QtCore.QRect(410, 240, 51, 21))
        self.label_10.setText("")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_5)
        self.label_11.setGeometry(QtCore.QRect(640, 210, 51, 21))
        self.label_11.setText("")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_5)
        self.label_12.setGeometry(QtCore.QRect(640, 180, 51, 21))
        self.label_12.setText("")
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_5)
        self.label_13.setGeometry(QtCore.QRect(640, 120, 51, 21))
        self.label_13.setText("")
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_5)
        self.label_14.setGeometry(QtCore.QRect(620, 40, 211, 31))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.tab_5)
        self.label_15.setGeometry(QtCore.QRect(640, 150, 51, 21))
        self.label_15.setText("")
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(self.tab_5)
        self.label_16.setGeometry(QtCore.QRect(640, 240, 51, 21))
        self.label_16.setText("")
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.tableView_2 = QtWidgets.QTableView(self.tab_6)
        self.tableView_2.setGeometry(QtCore.QRect(0, 70, 1101, 501))
        self.tableView_2.setObjectName("tableView_2")
        self.pushButton_12 = QtWidgets.QPushButton(self.tab_6)
        self.pushButton_12.setGeometry(QtCore.QRect(460, 20, 141, 31))
        self.pushButton_12.setObjectName("pushButton_12")
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_2.setGeometry(QtCore.QRect(30, 40, 391, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.comboBox_5 = QtWidgets.QComboBox(self.tab_7)
        self.comboBox_5.setGeometry(QtCore.QRect(470, 50, 151, 31))
        self.comboBox_5.setObjectName("comboBox_5")
        self.label_6 = QtWidgets.QLabel(self.tab_7)
        self.label_6.setGeometry(QtCore.QRect(70, 180, 241, 31))
        self.label_6.setObjectName("label_6")
        self.checkBox_13 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_13.setGeometry(QtCore.QRect(80, 310, 121, 17))
        self.checkBox_13.setObjectName("checkBox_13")
        self.checkBox_14 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_14.setGeometry(QtCore.QRect(80, 250, 111, 17))
        self.checkBox_14.setObjectName("checkBox_14")
        self.checkBox_15 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_15.setGeometry(QtCore.QRect(80, 230, 121, 17))
        self.checkBox_15.setObjectName("checkBox_15")
        self.checkBox_16 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_16.setGeometry(QtCore.QRect(80, 290, 141, 17))
        self.checkBox_16.setObjectName("checkBox_16")
        self.checkBox_17 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_17.setGeometry(QtCore.QRect(80, 330, 91, 17))
        self.checkBox_17.setObjectName("checkBox_17")
        self.pushButton_8 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_8.setGeometry(QtCore.QRect(500, 270, 101, 41))
        self.pushButton_8.setObjectName("pushButton_8")
        self.checkBox_18 = QtWidgets.QCheckBox(self.tab_7)
        self.checkBox_18.setGeometry(QtCore.QRect(80, 270, 91, 17))
        self.checkBox_18.setObjectName("checkBox_18")
        self.pushButton_9 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_9.setGeometry(QtCore.QRect(640, 120, 81, 31))
        self.pushButton_9.setObjectName("pushButton_9")
        self.comboBox_6 = QtWidgets.QComboBox(self.tab_7)
        self.comboBox_6.setGeometry(QtCore.QRect(470, 120, 151, 31))
        self.comboBox_6.setObjectName("comboBox_6")
        self.pushButton_10 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_10.setGeometry(QtCore.QRect(640, 50, 81, 31))
        self.pushButton_10.setObjectName("pushButton_10")
        self.tabWidget.addTab(self.tab_7, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tableView_3 = QtWidgets.QTableView(self.tab)
        self.tableView_3.setGeometry(QtCore.QRect(0, 100, 1091, 481))
        self.tableView_3.setObjectName("tableView_3")
        self.pushButton_11 = QtWidgets.QPushButton(self.tab)
        self.pushButton_11.setGeometry(QtCore.QRect(460, 30, 141, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.tabWidget.addTab(self.tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1095, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Browse"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Excel File Data"))
        self.label_2.setText(_translate("MainWindow", "Select Column Names"))
        self.pushButton_5.setText(_translate("MainWindow", "Add"))
        self.pushButton_6.setText(_translate("MainWindow", "Add"))
        self.label_3.setText(_translate("MainWindow", "Select Keyword Columns"))
        self.pushButton_7.setText(_translate("MainWindow", "See Selected Columns"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Select Column and Keyword"))
        self.checkBox.setText(_translate("MainWindow", "Logistic Regression"))
        self.checkBox_2.setText(_translate("MainWindow", "Linear SVC"))
        self.checkBox_3.setText(_translate("MainWindow", "Naive Bayes"))
        self.checkBox_4.setText(_translate("MainWindow", "Multinomial Naive Bayes"))
        self.checkBox_5.setText(_translate("MainWindow", "Decision Tree"))
        self.checkBox_6.setText(_translate("MainWindow", "Select All"))
        self.label_4.setText(_translate("MainWindow", "Select Machine Learning Algorithm For Training"))
        self.pushButton_3.setText(_translate("MainWindow", "Train"))
        self.label_5.setText(_translate("MainWindow", "Count Vectorizer Prediction Accuracy in %"))
        self.label_14.setText(_translate("MainWindow", "TF-IDF Vectorizer Prediction Accuracy in %"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Choose Model to Train"))
        self.pushButton_12.setText(_translate("MainWindow", "Show Train Results"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Train Results"))
        self.pushButton_2.setText(_translate("MainWindow", "Open File for Testing"))
        self.label_6.setText(_translate("MainWindow", "Select Machine Learning Algorithm For Testing"))
        self.checkBox_13.setText(_translate("MainWindow", "Decision Tree"))
        self.checkBox_14.setText(_translate("MainWindow", "Linear SVC"))
        self.checkBox_15.setText(_translate("MainWindow", "Logistic Regression"))
        self.checkBox_16.setText(_translate("MainWindow", "Multinomial Naive Bayes"))
        self.checkBox_17.setText(_translate("MainWindow", "Select All"))
        self.pushButton_8.setText(_translate("MainWindow", "Test"))
        self.checkBox_18.setText(_translate("MainWindow", "Naive Bayes"))
        self.pushButton_9.setText(_translate("MainWindow", "Add"))
        self.pushButton_10.setText(_translate("MainWindow", "Add"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "Choose Model to Test"))
        self.pushButton_11.setText(_translate("MainWindow", "Show Test Results"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Test Results"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())