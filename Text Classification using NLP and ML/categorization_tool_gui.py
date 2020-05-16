def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import os
import subprocess
import shlex
import platform
import csv
import re
import hickle as hkl
import pandas as pd
from numpy import dtype
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
pd.options.mode.chained_assignment = None


stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

from PyQt5.uic import loadUiType
# (?# ui, _ = loadUiType(r'C:\Users\d33ps3curity\Desktop\milestone_2\mainwindow.ui'))

from mainwindow import Ui_MainWindow

count = 0
class_len = 0
category_len = 0

test_data = None
class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.class_column_list = []
        self.category_column_list = []
        self.test_class_column_list = []
        self.test_keyword_column_list = []
        self.all_filenames = {}

        self.setupUi(self)
        self.comboBox_3.addItem('Select Class Column')
        self.comboBox_4.addItem('Select Keyword')
        self.comboBox_5.addItem('Select Column')
        self.comboBox_6.addItem('Select Keyword')
        self.handleButtons()

    def handleButtons(self):
        self.pushButton.clicked.connect(self.browseFile)
        self.pushButton_5.clicked.connect(self.addItem)
        self.pushButton_6.clicked.connect(self.addCategory)
        self.pushButton_10.clicked.connect(self.addTestItem)
        self.pushButton_9.clicked.connect(self.addTestCategory)
        self.pushButton_2.clicked.connect(self.browse_test_file)

    def startClassification(self, dataframe, keyword_col_name):
        data = dataframe
        filename = "preprocessed_" + keyword_col_name + ".xlsx"
        if os.path.exists(filename):
            keyword = pd.read_excel(filename)
            data[keyword_col_name] = keyword
        else:
            data = self.preprocess_data(dataframe, keyword_col_name)
        class_and_category_name = []
        for cls in self.class_column_list:
            for cat in self.category_column_list:
                class_and_category_name.append([cls, cat])

        classification = []
        for cls_and_cat in class_and_category_name:
            classification.append(data[cls_and_cat])


        model = []
        for i in range(len(classification)):
            model.append(pandasModel(classification[i]))

        if len(model) == 1:
            self.tableView_8.setModel(model[0])
        if len(model) == 2:
            self.tableView_8.setModel(model[0])
            self.tableView_6.setModel(model[1])
        if len(model) == 3:
            self.tableView_8.setModel(model[0])
            self.tableView_6.setModel(model[1])
            self.tableView_9.setModel(model[2])
        if len(model) == 4:
            self.tableView_8.setModel(model[0])
            self.tableView_6.setModel(model[1])
            self.tableView_9.setModel(model[2])
            self.tableView_7.setModel(model[3])

        self.pushButton_3.clicked.connect(lambda:self.train(data, class_and_category_name))


    def show_train_result(self, data):
        model = pandasModel(data)
        self.tableView_2.setModel(model)


    def train(self, data, class_and_category_name):

        predefined_classifier_list = ['Logistic Regression',
                                      'Linear SVC',
                                      'Multinomial Naive Bayes',
                                      'Decision Tree']
        classifier_list = []
        if not self.checkBox_6.isChecked():
            if self.checkBox.isChecked():
                classifier_list.append(self.checkBox.text())
            if self.checkBox_2.isChecked():
                classifier_list.append(self.checkBox_2.text())
            if self.checkBox_3.isChecked():
                classifier_list.append(self.checkBox_3.text())
            if self.checkBox_4.isChecked():
                classifier_list.append(self.checkBox_4.text())
            if self.checkBox_5.isChecked():
                classifier_list.append(self.checkBox_5.text())

        classifier_indices = []
        for selected_classifier in classifier_list:
            if selected_classifier in predefined_classifier_list:
                classifier_indices.append(predefined_classifier_list.index(selected_classifier))

        result = {}
        for i in classifier_indices:
            result[predefined_classifier_list[i]] = []

        if not self.checkBox_6.isChecked():
            for cls_and_column in class_and_category_name:
                for i in classifier_indices:
                    if predefined_classifier_list[i] == 'Logistic Regression':
                        classifier = LogisticRegression()
                        a, b, c, d, class_list = self.train_model(predefined_classifier_list[i], classifier, data[cls_and_column])
                        print('{} => accuracy_count: {}, accuracy_tfidf: {} '.format(predefined_classifier_list[i], a, c))
                        result[predefined_classifier_list[i]].append([[a, b], [c, d], class_list])
                        prediction_data = pd.DataFrame(list(zip(b, d)), columns=['count_pred', 'tfidf_pred'])
                        show_data = pd.concat([data[cls_and_column], prediction_data], axis=1)
                        self.label.setText(str(a)[:5])
                        self.label_13.setText(str(c)[:5])
                        self.pushButton_12.clicked.connect(lambda: self.show_train_result(show_data))
                    if predefined_classifier_list[i] == 'Linear SVC':
                        classifier = LinearSVC()
                        a, b, c, d, class_list = self.train_model(predefined_classifier_list[i], classifier, data[cls_and_column])
                        print('{} => accuracy_count: {}, accuracy_tfidf: {} '.format(predefined_classifier_list[i], a, c))
                        result[predefined_classifier_list[i]].append([[a, b], [c, d], class_list])
                        self.label_7.setText(str(a)[:5])
                        self.label_15.setText(str(c)[:5])
                    if predefined_classifier_list[i] == 'Multinomial Naive Bayes':
                        classifier = MultinomialNB()
                        a, b, c, d, class_list = self.train_model(predefined_classifier_list[i], classifier, data[cls_and_column])
                        print('{} => accuracy_count: {}, accuracy_tfidf: {} '.format(predefined_classifier_list[i], a, c))
                        result[predefined_classifier_list[i]].append([[a, b], [c, d], class_list])
                        self.label_9.setText(str(a)[:5])
                        self.label_11.setText(str(c)[:5])
                    if predefined_classifier_list[i] == 'Decision Tree':
                        classifier = DecisionTreeClassifier()
                        a, b, c, d, class_list = self.train_model(predefined_classifier_list[i], classifier, data[cls_and_column])
                        print('{} => accuracy_count: {}, accuracy_tfidf: {} '.format(predefined_classifier_list[i], a, c))
                        result[predefined_classifier_list[i]].append([[a, b], [c, d], class_list])
                        self.label_10.setText(str(a)[:5])
                        self.label_16.setText(str(c)[:5])

    def train_model(self, classifier_name, classifier, dataframe):
        columns_name = list(dataframe.columns)
        data = dataframe.dropna()
        count_vect = CountVectorizer()
        tfidf_vect = TfidfVectorizer()
        X_train, X_test, y_train, y_test = train_test_split(data[columns_name[1]],
                                                            data[columns_name[0]],
                                                            test_size=0.1,
                                                            random_state=100)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.fit_transform(y_test)
        encoder_class_filename = columns_name[0] + '_' + classifier_name + '_class_label.hkl'
        class_list = list(encoder.classes_)
        hkl.dump(class_list, encoder_class_filename)

        count_vect.fit(data[columns_name[1]])
        X_train_count = count_vect.transform(X_train)
        X_test_count = count_vect.transform(X_test)
        classifier.fit(X_train_count, y_train)

        predictions_count = classifier.predict(X_test_count)
        count_vect_model_filename = columns_name[0] + '_' + classifier_name + '_count' + '.hkl'
        count_vect_vectorized_filename = columns_name[0] + '_' + classifier_name + '_count_vectorized' + '.hkl'
        hkl.dump(classifier, count_vect_model_filename)
        hkl.dump(count_vect, count_vect_vectorized_filename)

        tfidf_vect.fit(data[columns_name[1]])
        X_train_tfidf = tfidf_vect.transform(X_train)
        X_test_tfidf = tfidf_vect.transform(X_test)
        classifier.fit(X_train_tfidf, y_train)

        tfidf_vect_model_filename = columns_name[0] + '_' + classifier_name + '_tfidf' + '.hkl'
        tfidf_vect_vectorized_filename = columns_name[0] + '_' + classifier_name + '_tfidf_vectorized' + '.hkl'
        hkl.dump(classifier, tfidf_vect_model_filename)
        hkl.dump(tfidf_vect, tfidf_vect_vectorized_filename)

        predictions_tfidf = classifier.predict(X_test_tfidf)

        accuracy_count = accuracy_score(predictions_count, y_test)
        accuracy_tfidf = accuracy_score(predictions_tfidf, y_test)

        self.all_filenames[columns_name[0] + '_' + classifier_name] = [count_vect_model_filename,
                                                            count_vect_vectorized_filename,
                                                            tfidf_vect_model_filename,
                                                            tfidf_vect_vectorized_filename,
                                                            encoder_class_filename]
        class_list = list(encoder.classes_)
        # print(class_list)
        accuracy_count = accuracy_count * 100
        accuracy_tfidf = accuracy_tfidf * 100
        predictions_count_class = [class_list[i] for i in predictions_count]
        predictions_tfidf_class = [class_list[i] for i in predictions_tfidf]
        return accuracy_count, predictions_count_class, accuracy_tfidf, predictions_tfidf_class, class_list

    def test(self, data, query_column_name):

        checked_classifier_list = []
        if not self.checkBox_17.isChecked():
            if self.checkBox_15.isChecked():
                checked_classifier_list.append(str(self.checkBox_15.text()))
            if self.checkBox_14.isChecked():
                checked_classifier_list.append(str(self.checkBox_14.text()))
            if self.checkBox_18.isChecked():
                checked_classifier_list.append(str(self.checkBox_18.text()))
            if self.checkBox_16.isChecked():
                checked_classifier_list.append(str(self.checkBox_16.text()))
            if self.checkBox_13.isChecked():
                checked_classifier_list.append(self.checkBox_13.text())
        else:
            checked_classifier_list.append(self.checkBox_17.text())

        saved_models = ['Logistic Regression',
                        'Linear SVC',
                        'Multinomial Naive Bayes',
                        'Decision Tree']
        saved_filenames = {}
        for saved_model in saved_models:
            saved_filenames[saved_model] = []

        for checked_classifier in checked_classifier_list:
            for class_name in self.test_class_column_list:
                saved_filenames[checked_classifier].append(class_name + '_' + checked_classifier)

        filename_list = os.listdir()
        show_data = []
        for key, value in saved_filenames.items():
            for name in value:
                test_filename_list = []
                for filename in filename_list:
                    if re.search(name, filename):
                        test_filename_list.append(filename)
                show_data.append(self.load_saved_model(test_filename_list[0],
                                      test_filename_list[1],
                                      test_filename_list[2],
                                      name,
                                      data[query_column_name]))
                show_data.append(self.load_saved_model(test_filename_list[0],
                                      test_filename_list[3],
                                      test_filename_list[4],
                                      name,
                                      data[query_column_name]))

        self.pushButton_11.clicked.connect(lambda: self.view_result(show_data))

    def view_result(self, show_data):

        show_data = pd.merge(show_data[0], show_data[1], on='keyword')
        model_1 = pandasModel(show_data)
        self.tableView_3.setModel(model_1)

    def load_saved_model(self, labeled_class_name, model_filename, vect_filename,cat_name, data):

        if os.path.exists(model_filename):
            loaded_model = hkl.load(model_filename)
        if os.path.exists(vect_filename):
            loaded_vectorizer = hkl.load(vect_filename)

        test_data = list(data)
        labeled_class_list = hkl.load(labeled_class_name)

        X_val_vec = loaded_vectorizer.transform(test_data)
        prediction = loaded_model.predict(X_val_vec)
        prediction_list = [labeled_class_list[prediction[idx]] for idx in prediction]

        # print(test_data[:5])
        # print(prediction_list[:5])
        data_tuples = list(zip(prediction_list, test_data))
        return_data = pd.DataFrame(data_tuples, columns=['vectorizer', 'keyword'])

        return return_data

    def addTestItem(self):
        item_name = str(self.comboBox_5.currentText())
        self.test_class_column_list.append(item_name)
        print(self.test_class_column_list)
    def addTestCategory(self):
        category_name = str(self.comboBox_6.currentText())
        self.test_keyword_column_list.append(category_name)
        print(self.test_keyword_column_list)

    def browse_test_file(self):
        url = QFileDialog.getOpenFileName(self, 'Open a File', '', 'All Files(*);;*xlsx')
        file_url = url[0]
        data = pd.read_excel(file_url)
        # data = data
        show_data = data.head(200)
        # data = data.head()
        model = pandasModel(show_data)
        self.tableView.setModel(model)
        column_names = list(data.columns)
        for column_name in column_names:
            if 'keyword' in column_name.lower() or 'query' in column_name.lower():
                self.comboBox_6.addItem(column_name)
                keyword_col_name = column_name
            else:
            	if data[column_name].dtype == dtype:
                	self.comboBox_5.addItem(column_name)

        self.pushButton_8.clicked.connect(lambda: self.test(data, keyword_col_name))



    def browseFile(self):
        url = QFileDialog.getOpenFileName(self, 'Open a File', '', 'All Files(*);;*xlsx')
        file_url = url[0]
        data = pd.read_excel(file_url)
        show_data = data.head(42)
        model = pandasModel(show_data)
        self.tableView.setModel(model)
        temp_column_names = list(data.columns)
        column_names = []
        for col in temp_column_names:
            if data[col].dtype == dtype:
                column_names.append(col)
        for column_name in column_names:
            if 'keyword' not in column_name.lower():
                self.comboBox_3.addItem(column_name)
                # self.comboBox_5.addItem(column_name)
            else:
                self.comboBox_4.addItem(column_name)
                keyword_col_name = column_name

        self.pushButton_7.clicked.connect(lambda: self.startClassification(data, keyword_col_name))

    def addItem(self):
        global class_len
        class_len += 1
        item_name = str(self.comboBox_3.currentText())
        self.class_column_list.append(item_name)
        print(self.class_column_list)

    def addCategory(self):
        global category_len
        category_len += 1
        category_name = str(self.comboBox_4.currentText())
        self.category_column_list.append(category_name)
        print(self.category_column_list)

    def preprocess_data(self, dataframe, keyword_col_name):
        print(keyword_col_name)
        data = dataframe
        filename = 'preprocessed_' + keyword_col_name + '.xlsx'
        if data.isnull().sum().sum():
            data = data.dropna()
        keyword = [text for text in data[keyword_col_name]]
        keyword = [self.preprocessing(text) for text in keyword]
        keyword = pd.DataFrame(keyword)
        keyword = keyword.rename(columns={list(keyword.columns)[0]: keyword_col_name})
        keyword.to_excel(filename, index=None)
        data[keyword_col_name] = keyword
        # print('In Preproces_data function\n')
        # print(data.head())
        return data

    def preprocessing(self, document):
        global stopwords
        document = str(document)
        document = document.lower()
        document = re.sub('\W\D', ' ', document)
        words = document.split()
        words = [word for word in words if word not in stopwords]
        document = str(' '.join(words))
        return document


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
