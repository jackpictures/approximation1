from PyQt5 import uic, QtWidgets
from PyQt5 import QtCore, QtGui
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import func as f
import module1 as md


Form, _ = uic.loadUiType("1.ui")
Form1, _ =uic.loadUiType("gen.ui")
Form2, _ =uic.loadUiType("pol.ui")
Form3, _ =uic.loadUiType("alpha.ui")
Form4, _ = uic.loadUiType("batch.ui")
Form5, _ = uic.loadUiType("Qbatch.ui")


class EmittingStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class Qbatch(QtWidgets.QDialog, Form5):
    def __init__(self,root):
        super(Qbatch, self).__init__(root)
        self.main = root
        self.setupUi(self)
        self.setWindowTitle(' ')
        self.buttonBox.accepted.connect(self.Graph)
        self.buttonBox.rejected.connect(self.Close)

    def Graph(self):
        plt.figure(5).clf()
        plt.figure(5).clear()
        plt.figure(5)
        t_ = []
        loss_ = []
        iter_ = []
        alpha = float(self.lineEdit_6.text())
        plt.xlim(float(self.lineEdit.text()), float(self.lineEdit_3.text()))
        plt.ylim(float(self.lineEdit_2.text()), float(self.lineEdit_4.text()))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        mb = int(self.lineEdit_7.text())
        step = int(self.lineEdit_8.text())
        for i in range(int(self.lineEdit_5.text())):
            t, loss, iter = f.mb_gradient_descent(alpha, self.main.x, self.main.y, self.main.thetas,
                                                  mb, 0.01, 30)
            t_.append(t)
            loss_.append(loss)
            it = []
            for j in range(iter):
                it.append(j)
            iter_.append(it)
            c = colors[i] + 'o-'
            s = 'minibatch = ' + str(mb)
            mb += step
            plt.plot(iter_[i], loss_[i], c, label=s)
        plt.xlabel("Iteration")
        plt.ylabel("Q(θ)")
        self.close()
        plt.legend()
        plt.show()

    def Close(self):
        self.close()

class batch(QtWidgets.QDialog, Form4):
    def __init__(self,root):
        super(batch, self).__init__(root)
        self.main = root
        self.setupUi(self)
        self.setWindowTitle(' ')
        self.buttonBox.accepted.connect(self.Graph)
        self.buttonBox.rejected.connect(self.Close)

    def Graph(self):
        plt.figure(4).clf()
        plt.figure(4).clear()
        plt.figure(4)
        mb = int(self.lineEdit.text())
        step = int(self.lineEdit_2.text())
        batch = []
        it = []
        for i in range((self.main.x.shape[0]-mb)//step+1):
            t, loss, iter = f.mb_gradient_descent(float(self.main.lineEdit_6.text()), self.main.x, self.main.y,
                                                                 self.main.thetas, mb, float(self.main.lineEdit_7.text()),
                                                                 int(self.main.lineEdit_9.text()))
            batch.append(mb)
            it.append(iter)
            mb+=step
        plt.plot(batch,it,'bo-')
        plt.xlabel("Batch size")
        plt.ylabel("Iteration")
        self.close()
        plt.show()
    def Close(self):
        self.close()

class alpha(QtWidgets.QDialog, Form3):
    def __init__(self,root):
        super(alpha, self).__init__(root)
        self.main = root
        self.setupUi(self)
        self.setWindowTitle(' ')
        self.buttonBox.accepted.connect(self.Graph)
        self.buttonBox.rejected.connect(self.Close)

    def Graph(self):
        plt.figure(3).clf()
        plt.figure(3).clear()
        plt.figure(3)
        t_ = []
        loss_ = []
        iter_ = []
        alpha = float(self.lineEdit_6.text())
        plt.xlim(float(self.lineEdit.text()), float(self.lineEdit_3.text()))
        plt.ylim(float(self.lineEdit_2.text()), float(self.lineEdit_4.text()))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(int(self.lineEdit_5.text())):
            if (self.main.comboBox.currentText()=="Vanilla Gradient Descent"):
                t, loss, iter = f.gradient_descent(alpha, self.main.x, self.main.y, self.main.thetas, 0.01, 30)
            if (self.main.comboBox.currentText() == "Stochastic Gradient Descent"):
                t, loss, iter = f.st_gradient_descent(alpha, self.main.x, self.main.y, self.main.thetas, 0.01, 30)
            if (self.main.comboBox.currentText() == "Mini-batch Gradient Descent"):
                t, loss, iter = f.mb_gradient_descent(alpha, self.main.x, self.main.y, self.main.thetas,int(self.main.lineEdit.text()), 0.01, 30)
            print(alpha)
            t_.append(t)
            loss_.append(loss)
            it = []
            for j in range(iter):
                it.append(j)
            iter_.append(it)
            alpha += float(self.lineEdit_7.text())
            c = colors[i] + 'o-'
            s = 'alpha = ' + str(alpha)
            plt.plot(iter_[i], loss_[i], c, label=s)
        plt.xlabel("Iteration")
        plt.ylabel("Q(θ)")
        self.close()
        plt.legend()
        plt.show()

    def Close(self):
        self.close()

class pol(QtWidgets.QDialog, Form2):
    def __init__(self,root):
        super(pol, self).__init__(root)
        self.main = root
        self.setupUi(self)
        self.setWindowTitle(' ')
        self.pushButton.clicked.connect(self.Changed)
        self.pushButton_2.clicked.connect(self.Click)
        self.pushButton_2.hide()

    def Changed(self):
        self.pushButton_2.show()
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["θ_i","значение"])
        self.tableWidget.setRowCount(int(self.lineEdit.text())+2)
        self.tableWidget.verticalHeader().hide()
        for i in range(0,int(self.lineEdit.text())+1):
            item = QtWidgets.QTableWidgetItem("θ_%s=" % i)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tableWidget.setItem(i+1, 0 , item)

    def Click(self):
        self.main.pushButton_3.setEnabled(False)
        self.main.thetas = []
        for i in range(1,self.tableWidget.rowCount()):
            self.main.thetas.append(float(self.tableWidget.item(i,1).text()))
        self.main.pushButton.setEnabled(True)
        st = md.print_poly((self.main.thetas))
        if (st[0] == '+'):
            st = ((st[1:]))
        self.main.label_4.setText("θ = %s\nF(θ,x) = %s" % (self.main.thetas,st))
        self.main.label_3.setText("")
        self.main.pushButton.setStyleSheet("background-color: rgb(62,207,137)")
        self.main.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.close()


class Gen(QtWidgets.QDialog, Form1):
    def __init__(self, root):
        super(Gen, self).__init__(root)
        self.main = root
        self.setupUi(self)
        self.setWindowTitle(" ")
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).setText("Отмена")
        self.buttonBox.accepted.connect(self.close_window)
        self.buttonBox.rejected.connect(self.close_window1)

    def close_window(self):
        if (self.main.thetas ==[]):
            print("Enter thetas!")
            self.close()
        else:
            self.main.x, self.main.y = make_regression(n_samples=int(self.lineEdit.text()), n_features=1,
                                                       n_informative=1,
                                                       random_state=0, noise=0)
            self.main.pushButton.setStyleSheet("background-color:")

            for i in range(len(self.main.y)):
                self.main.y[i] = 0
            for i in range(len(self.main.y)):
                for j in range(len(self.main.thetas)):
                    self.main.y[i] += float(self.main.thetas[j]) * (float(self.main.x[i][0]) ** j)

            #print(self.main.thetas)
            self.main.ytest = self.main.y
            self.main.xtest = self.main.x
            self.main.y += np.random.normal(int(self.lineEdit_4.text()), int(self.lineEdit_5.text()),
                                            size=self.main.y.shape)
            #self.main.xtrain, self.main.xtest = train_test_split(self.main.x, test_size=float(self.lineEdit_2.text()),
              #                                                  random_state=int(self.lineEdit_3.text()))
            #self.main.ytrain, self.main.ytest = train_test_split(self.main.y, test_size=float(self.lineEdit_2.text()),
             #                                                    random_state=int(self.lineEdit_3.text()))
            self.main.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
            self.main.label_3.setText("Количество = %s\nСреднеквадратичное отклонение = %s\nДисперсия = %s\n" % (int(self.lineEdit.text()),int(self.lineEdit_4.text()),
                                                                                      int(self.lineEdit_5.text())))
            self.main.pushButton_4.setEnabled(True)
            self.main.pushButton_5.setEnabled(True)
            self.main.pushButton_6.setEnabled(True)
            self.close()

    def close_window1(self):
        self.close()


class Ui(QtWidgets.QDialog, Form):
    def __init__(self):
        super(Ui, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Аппроксимация")
        self.setWindowIcon(QtGui.QIcon('pngwing.com.ico'))
        #инициализация подклассов
        self.gen = Gen(self)
        self.p = pol(self)
        self.a = alpha(self)
        self.b = batch(self)
        self.q = Qbatch(self)
        #инициализация переменных
        self.loss = 0
        self.iter = 0
        self.thetas = []
        self.t = []
        self.x = []
        self.y = []
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Cancel).setText("Выход")
        self.pushButton_5.hide()
        self.pushButton_6.hide()
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        self.pushButton.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.lineEdit.hide()
        self.label_2.hide()

        self.pushButton.clicked.connect(self.gen.exec)
        self.pushButton_2.clicked.connect(self.p.exec)
        self.pushButton_4.clicked.connect(self.a.exec)
        self.pushButton_5.clicked.connect(self.b.exec)
        self.pushButton_6.clicked.connect(self.q.exec)
        self.buttonBox.accepted.connect(self.Approximation)
        self.buttonBox.rejected.connect(self.Close)
        self.pushButton_3.clicked.connect(self.Graphs)
        self.comboBox.currentTextChanged.connect(self.cbchanged)


    def __del__(self):
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()


    def cbchanged(self):
        self.pushButton_3.setEnabled(False)
        if (self.comboBox.currentText() == "Mini-batch Gradient Descent"):
            self.lineEdit.show()
            self.label_2.show()
            self.pushButton_5.show()
            self.pushButton_6.show()
        else:
            self.lineEdit.hide()
            self.label_2.hide()
            self.pushButton_5.hide()
            self.pushButton_6.hide()

    def Approximation(self):
        #print(self.x,self.y)
        self.pushButton_3.setEnabled(True)
        if (self.comboBox.currentText()=="Vanilla Gradient Descent"):
            self.t, self.loss,self.iter = f.gradient_descent(float(self.lineEdit_6.text()), self.x, self.y,self.thetas
                                              , float(self.lineEdit_7.text()),int(self.lineEdit_9.text()))
            if len(self.t) != 0:
                x1 = self.x
                y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
                for i in range(len(y_predict)):
                    y_predict[i] = 0
                for i in range(len(y_predict)):
                    for j in range(len(self.t)):
                        y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
                print('θ = %s\n Q(θ) = %s\nТочность = %s' % (self.t, self.loss[-1], (f.score(self.y, y_predict))))
            else:
                print("Try another alpha")
        if (self.comboBox.currentText() == "Stochastic Gradient Descent"):
            self.t, self.loss, self.iter = f.st_gradient_descent(float(self.lineEdit_6.text()), self.x, self.y,self.thetas
                                              , float(self.lineEdit_7.text()),int(self.lineEdit_9.text()))
            if len(self.t) != 0:
                x1 = self.x
                y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
                for i in range(len(y_predict)):
                    y_predict[i] = 0
                for i in range(len(y_predict)):
                    for j in range(len(self.t)):
                        y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
                print('θ = %s\n Q(θ) = %s\nТочность = %s' % (self.t, self.loss[-1], (f.score(self.y, y_predict))))
            else:
                print("Try another alpha")

        if (self.comboBox.currentText() == "Mini-batch Gradient Descent"):
            batch_size = int(self.lineEdit.text())
            self.t, self.loss, self.iter = f.mb_gradient_descent(float(self.lineEdit_6.text()), self.x, self.y,self.thetas, batch_size, float(self.lineEdit_7.text()),int(self.lineEdit_9.text()))
            if len(self.t)!=0:
                x1 = self.x
                y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
                for i in range(len(y_predict)):
                    y_predict[i] = 0
                for i in range(len(y_predict)):
                    for j in range(len(self.t)):
                        y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
                print('θ = %s\n Q(θ) = %s\nТочность = %s' % (self.t,self.loss[-1],(f.score(self.y, y_predict))))
            else:
                print("Try another alpha")

        if self.comboBox.currentText() == "Sklearn SGDRegressor":
            poly_features = PolynomialFeatures(degree=(len(self.thetas) - 1), include_bias=False)
            x_poly = poly_features.fit_transform(self.x)
            self.t, sgd, self.iter = f.ls_sklearn_sgd(x_poly, self.y,int(self.lineEdit_9.text()),float(self.lineEdit_6.text()),float(self.lineEdit_7.text()))
            self.t = list(map(float,self.t))
            print("Converged, iterations: %s" % self.iter)
            print("θ = %s\n Q(θ) = %s\nТочность = %s" % (self.t,1-sgd.score(x_poly,self.y),sgd.score(x_poly,self.y)))



    def Graphs(self):
        plt.figure(1).clf()
        plt.figure(1).clear()
        plt.figure(2).clf()
        plt.figure(2).clear()
        if (self.comboBox.currentText()=="Vanilla Gradient Descent"):
            plt.figure(1)
            it = []
            for i in range(self.iter):
                it.append(i)
            plt.plot(it, self.loss,'bo-')
            plt.title('Vanilla Gradient Descent loss')
            plt.xlabel('Iteration')
            plt.ylabel('Q(θ)')
            plt.figure(2)
            plt.plot(self.x, self.y, 'go', label='dataset')
            x1 = sorted(self.x)
            y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
            for i in range(len(y_predict)):
                y_predict[i] = 0
            for i in range(len(y_predict)):
                for j in range(len(self.t)):
                    y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
            plt.plot(x1, y_predict, '-r', label='result')
            plt.title('Vanilla Gradient Descent')
                #plt.plot(iter_[i], loss_[i], 'ro-', label='alpha = 0.31')
                #plt.plot(iter_[i], loss_[i], 'yo-', label='alpha = 0.61')
            #plt.plot(iter_[3], loss_[3], 'go-', label='alpha = 0.91')
            plt.legend()
            plt.show()
        if (self.comboBox.currentText() == "Stochastic Gradient Descent"):
            plt.figure(1)
            it = []
            for i in range(self.iter):
                it.append(i)
            plt.plot(it, self.loss)
            plt.title('Stochastic Gradient Descent loss')
            plt.xlabel('Iteration')
            plt.ylabel('Q(θ)')
            plt.figure(2)
            plt.plot(self.x, self.y, 'go', label='dataset')
            x1 = sorted(self.x)
            y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
            for i in range(len(y_predict)):
                y_predict[i] = 0
            for i in range(len(y_predict)):
                for j in range(len(self.t)):
                    y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
            plt.plot(x1, y_predict, '-r', label='result')
            plt.title('Stochastic Gradient Descent')
            plt.legend()
            plt.show()
        if (self.comboBox.currentText() == "Mini-batch Gradient Descent"):
            plt.figure(1)
            it = []
            for i in range(self.iter):
                it.append(i)
            plt.plot(it, self.loss)
            plt.title('Mini-batch Gradient Descent loss')
            plt.xlabel('Iteration')
            plt.ylabel('Q(θ)')
            plt.figure(2)
            plt.plot(self.x, self.y, 'go', label='dataset')
            x1 = sorted(self.x)
            y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
            for i in range(len(y_predict)):
                y_predict[i] = 0
            for i in range(len(y_predict)):
                for j in range(len(self.t)):
                    y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
            plt.plot(x1, y_predict, '-r', label='result')
            plt.title('Mini-batch Gradient Descent')
            plt.legend()
            plt.show()
        if self.comboBox.currentText() == "Sklearn SGDRegressor":
            plt.figure(1)
            plt.plot(self.x, self.y, 'go', label='dataset')
            x1 = sorted(self.x)
            y_predict = np.ndarray(shape=(self.y.shape), dtype=float)
            for i in range(len(y_predict)):
                y_predict[i] = 0
            for i in range(len(y_predict)):
                for j in range(len(self.t)):
                    y_predict[i] += float(self.t[j]) * (float(x1[i][0]) ** j)
            plt.plot(x1, y_predict, '-r', label='predict')
            plt.title('Sklearn SGDRegressor')
            plt.legend()
            plt.figure(1).show()

    def Close(self):
        self.close()




if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Ui()
    w.show()  # show window
    sys.exit(app.exec_())