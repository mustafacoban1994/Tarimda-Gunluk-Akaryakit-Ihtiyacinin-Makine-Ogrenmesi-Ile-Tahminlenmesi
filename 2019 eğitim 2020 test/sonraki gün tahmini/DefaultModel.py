import sys
import datetime
import pandas as pd
import cx_Oracle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pandas_datareader import data
import re
import time
from datetime import date
#import holidays
from sklearn import metrics
from matplotlib import pyplot
from sklearn.datasets import make_regression #feature importance icin gerekli
from sklearn.datasets import make_classification #feature importance icin gerekli

dfDb = pd.read_excel('veri.xlsx', sheet_name='Sheet1')


df = pd.read_excel('yagis_petrol.xlsx', sheet_name='antalya')

# df_new = pd.merge(dfDb, df, on='tarih', how='outer')



dates = df.iloc[:, 0:1]
datesValues = dates.values.tolist()
countDayWeek = 0
daysOfWeek = []
while countDayWeek < len(dates):
    daysOfWeek.append((pd.Timestamp(datesValues[countDayWeek][
                                        0]).dayofweek))  # GUN ADI eklemek icin , pd.Timestamp(abc[count][0]).day_name son guncellemede day_name calismamaktadir.
    countDayWeek = countDayWeek + 1  # 0 = pazartesi

countDayYear = 0
daysOfYear = []
while countDayYear < len(dates):
    daysOfYear.append((pd.Timestamp(datesValues[countDayYear][0]).dayofyear))  # Yilinkacinci gunu
    countDayYear = countDayYear + 1

countDayMonth = 0
daysOfMonth = []
while countDayMonth < len(dates):
    daysOfMonth.append((pd.Timestamp(datesValues[countDayMonth][0]).day))  # Ayin kacinin gunu
    countDayMonth = countDayMonth + 1

# ayın ilk günü
firstdayOfMonth = []
for firstday in daysOfMonth:
    if firstday == 1:
        firstdayOfMonth.append(1)
    else:
        firstdayOfMonth.append(0)

weekendOfWeek = []
for weekend in daysOfWeek:
    if weekend == 5 or weekend == 6:
        weekendOfWeek.append(1)
    else:
        weekendOfWeek.append(0)

# tatiller
holidays = ['2019-01-01', '2019-04-23', '2019-05-01', '2019-05-19',
            '2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', 
            '2019-07-15', '2019-08-10', '2019-08-11', '2019-08-12',
            '2019-08-13', '2019-08-14', '2019-08-30', '2019-10-28',
            '2019-10-29',
            
            '2020-01-01', '2020-04-23', '2020-05-01', '2020-05-19',
            '2020-05-23', '2020-05-24', '2020-05-25', '2020-05-26',
            '2020-07-15', '2020-07-30', '2020-07-31', '2020-08-01',
            '2020-08-02', '2020-08-03', '2020-08-30', '2020-10-28',
            '2020-10-29',
            '2021-01-01'
            ]

holidaysDatetime = []
for i in range(len(holidays)):
    holidaysDatetime.append(datetime.datetime.strptime(holidays[i], '%Y-%m-%d'))  # convert string to date
    
dfHolidays = pd.DataFrame(holidaysDatetime, columns=['Value'])
dfHolidays.columns = ['tarih']

dfDaysOfMonth = pd.DataFrame(daysOfMonth, columns=['gunler_ay'])
dfDaysOfWeek = pd.DataFrame(daysOfWeek, columns=['gunler_hafta'])
dfDaysOfYear = pd.DataFrame(daysOfYear, columns=['gunler_yil'])
dfFirstOfMonth = pd.DataFrame(firstdayOfMonth, columns=['ilkgun_ay'])
dfWeekendOfWeek = pd.DataFrame(weekendOfWeek, columns=['haftasonu'])
allData = pd.concat([df, dfDaysOfMonth, dfDaysOfWeek, dfDaysOfYear, dfFirstOfMonth, dfWeekendOfWeek], axis=1)

allData["holiday"] = allData["tarih"].isin(dfHolidays["tarih"])

hol = []
for holDay in allData["holiday"]:
    if holDay == True:
        hol.append(1)
    else:
        hol.append(0)

allData["holiday"] = hol

pivot_table = dfDb.pivot_table(index=["tarih"], columns=["uye_no"], values="miktar").fillna(0)
dfData = allData.set_index("tarih").join(pivot_table, lsuffix='_caller', rsuffix='_other', on='tarih').fillna(0)


# print (allData['tarih'].dtype) tip kontrol

# keras ile cnn modeli kurup bir sonraki günün tahminini 


litre = dfData.iloc[:, 1:2].values
y = dfData.iloc[:, 0:1].values
x = dfData.iloc[:, 2:].values
x = pd.DataFrame(x)
litre = pd.DataFrame(litre)
litreShift = litre.shift(periods = 1)
litreShift = litreShift.fillna(0)
y = pd.DataFrame(y)
yNormal = y
yShift = y.shift(periods=-1)
yShift = yShift.fillna(0)
y = round(yShift)


#xtrain 364 olacak
#xtest 365 olacak
#y train 364 olacak
#ytest 365 olacak
startPoint = 0
splint = 365  
#731 2021' e kadar
lastNum = len(x) - splint
#731 2021' e kadar
x_train = x.iloc[0:splint, :].values
x_test = x.iloc[splint:, :].values
y_train = y.iloc[0:splint, :].values
y_test = y.iloc[splint:, :].values


dfXTest = pd.DataFrame(x_test)
litreXTest = dfXTest.iloc[:,4:5]

xForDtrClass = pd.read_excel('xForDtrClassYeni.xlsx')
xForDtrReg = pd.read_excel('xForDtrRegYeni.xlsx')
xForKnn = pd.read_excel('xForKnnYeni.xlsx')
xForLogisticRegression = pd.read_excel('xForLogisticRegressionYeni.xlsx')
xForNn = pd.read_excel('xForNnYeni.xlsx')
xForRfc = pd.read_excel('xForRfcYeni.xlsx')
xForSvm = pd.read_excel('xForSvmYeni.xlsx')
xForXgb = pd.read_excel('xForXgbYeni.xlsx')

x_train_impDtc = xForDtrClass.iloc[0:splint, :].values
x_test_impDtc = xForDtrClass.iloc[splint:, :].values

x_train_impDtr = xForDtrReg.iloc[0:splint, :].values
x_test_impDtr = xForDtrReg.iloc[splint:, :].values

x_train_impKnn = xForKnn.iloc[0:splint, :].values
x_test_impKnn = xForKnn.iloc[splint:, :].values

x_train_impLr = xForLogisticRegression.iloc[0:splint, :].values
x_test_impLr = xForLogisticRegression.iloc[splint:, :].values

x_train_impNn = xForNn.iloc[0:splint, :].values
x_test_impNn = xForNn.iloc[splint:, :].values

x_train_impRfc = xForRfc.iloc[0:splint, :].values
x_test_impRfc = xForRfc.iloc[splint:, :].values

x_train_impSvm = xForSvm.iloc[0:splint, :].values
x_test_impSvm = xForSvm.iloc[splint:, :].values

x_train_impXgb = xForXgb.iloc[0:splint, :].values
x_test_impXgb = xForXgb.iloc[splint:, :].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(x_train)
X_test_sc = sc.transform(x_test) #fit_transform: x_trainden ogren ve transform et, tek transform ogrenmeden kullan demek


def KnnPred (xTrain,xTest,yTrain):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xTrainSc = sc.fit_transform(xTrain)
    xTestSc = sc.transform(xTest) #fit_transform: x_trainden ogren ve transform et, tek transform ogrenmeden kullan demek
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski') #n_neighbors komsu sayisi, metric, mesafe olcumu
    model.fit(xTrainSc, yTrain) #fit edemiyor, y' dekis ayılar float olduğu için fit edemediği farkeidlid ve sayılar yuvarlandı
    yPred = model.predict(xTestSc)
    return yPred



yPredKnn = KnnPred(x_train,x_test,y_train)
yPredKnnImportance = KnnPred(x_train_impKnn,x_test_impKnn,y_train)



def DtrRegPred(xTrain,xTest,yTrain):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=0)
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    return yPred


yPredDtr = DtrRegPred(x_train,x_test,y_train)
yPredImportanceDtr = DtrRegPred(x_train_impDtr,x_test_impDtr,y_train)



def DtrClassPred(xTrain,xTest,yTrain):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    return yPred


yPredDtrClass = DtrClassPred(x_train,x_test,y_train)
yPredImportanceDtrClass = DtrClassPred(x_train_impDtc,x_test_impDtc,y_train)


def LogisticRegressionPred(xTrain,xTest,yTrain):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred


yPredLogisticRegression = LogisticRegressionPred(x_train,x_test,y_train)
yPredImportanceLogisticRegression = LogisticRegressionPred(x_train_impLr,x_test_impLr,y_train)


def SvmPred(xTrain,xTest,yTrain):
    from sklearn import svm
    model = svm.LinearSVC()
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred



yPredSvm = SvmPred(x_train,x_test,y_train)
yPredImportanceSvm = SvmPred(x_train_impSvm,x_test_impSvm,y_train)


def RfclassPred(xTrain,xTest,yTrain):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred


yPredRfc = RfclassPred(x_train,x_test,y_train)
yPredImportanceRfc = RfclassPred(x_train_impRfc,x_test_impRfc,y_train)



def NnPred(xTrain,xTest,yTrain):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred


yPredNn = NnPred(x_train,x_test,y_train)
yPredImportanceNn = NnPred(x_train_impNn,x_test_impNn,y_train)


def XgbPred(xTrain,xTest,yTrain):
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred


def XgbHingePred(xTrain,xTest,yTrain):
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="binary:hinge")
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    return yPred



yPredXgb = XgbPred(x_train,x_test,y_train)
yPredImportanceXgb = XgbPred(x_train_impXgb,x_test_impXgb,y_train)
yPredXgbHinge = XgbHingePred(x_train,x_test,y_train)
yPredImportanceXgbHinge = XgbHingePred(x_train_impXgb,x_test_impXgb,y_train)


models = ['Knn - K-Nearest Neighbors ','Knn - K-Nearest Neighbors Importance',
          'Dtr - Decision Tree Regression', 'Dtr - Decision Tree Regression Importance',
          'Dtr-Classifier - Decision Tree Classifier','Dtr-Classifier - Decision Tree Classifier Importance',
          'LR - Logistic Regression','LR - Logistic Regression Importance',
          'Svm - Support Vector Machine','Svm - Support Vector Machine Importance',
          'RfClass - Random Forest Class','RfClass - Random Forest Class Importance',
          'Nn - Neural Network','Nn - Neural Network Importance',
          'XgbCls','XgbCls Importance',
          'XgbClsHinge','XgbClsHinge Importance']

predModels = [yPredKnn,yPredKnnImportance,
              yPredDtr,yPredImportanceDtr,
              yPredDtrClass,yPredImportanceDtrClass,
              yPredLogisticRegression,yPredImportanceLogisticRegression,
              yPredSvm,yPredImportanceSvm,
              yPredRfc,yPredImportanceRfc,
              yPredNn,yPredImportanceNn,
              yPredXgb,yPredImportanceXgb,
              yPredXgbHinge,yPredImportanceXgbHinge]
predModels = pd.DataFrame(predModels)
predModels = predModels.transpose()

r2Scores = []
accScores = []
mseScores = []

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

for i in range(len(models)):
    accScores.append('%s, %.5f' % (models[i],(accuracy_score(y_test, predModels[i]))))
    r2Scores.append('%s, %.5f' % (models[i],(r2_score(y_test, predModels[i]))))
    mseScores.append('%s, %.5f' % (models[i],(mean_squared_error(y_test, predModels[i]))))
    #confusionMatrix.append('%s, ConfusionMatrix: %.5f' % (models[i],(confusion_matrix(y_test, predModels[i]))))

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

f1Scores = []
recallScores = []
precisionScores = []

averages = ['macro','micro','weighted', 'binary']
for i in range(len(models)):
    for j in range(len(averages)):
        f1Scores.append('%s, f1Score:%s %.5f' % (models[i],averages[j],(f1_score(y_test, predModels[i], average=averages[j]))))
        recallScores.append('%s, recallScores:%s %.5f' % (models[i],averages[j],(recall_score(y_test, predModels[i], average=averages[j]))))
        precisionScores.append('%s, precisionScores:%s %.5f' % (models[i],averages[j],(precision_score(y_test, predModels[i], average=averages[j]))))

from sklearn.metrics import confusion_matrix
ConfusionKnn = confusion_matrix(y_test, yPredKnn)
ConfusionKnnImportance = confusion_matrix(y_test, yPredKnnImportance)
ConfusionDtr = confusion_matrix(y_test, yPredDtr)
ConfusionDtrImportance = confusion_matrix(y_test, yPredImportanceDtr)
ConfusionDtClass = confusion_matrix(y_test, yPredDtrClass)
ConfusionDtClassImportance = confusion_matrix(y_test, yPredImportanceDtrClass)
ConfusionLR = confusion_matrix(y_test, yPredLogisticRegression)
ConfusionLRImportance = confusion_matrix(y_test, yPredImportanceLogisticRegression)
ConfusionSvm = confusion_matrix(y_test, yPredSvm)
ConfusionSvmImportance = confusion_matrix(y_test, yPredImportanceSvm)
ConfusionRf = confusion_matrix(y_test, yPredRfc)
ConfusionRfImportance = confusion_matrix(y_test, yPredImportanceRfc)
ConfusionNn = confusion_matrix(y_test, yPredNn)
ConfusionNnImportance = confusion_matrix(y_test, yPredImportanceNn)
ConfusionXgbCls = confusion_matrix(y_test, yPredXgb)
ConfusionXgbClsImportance = confusion_matrix(y_test, yPredImportanceXgb)
ConfusionXgbClsHinge = confusion_matrix(y_test, yPredXgbHinge)
ConfusionXgbClsHingeImportance = confusion_matrix(y_test, yPredImportanceXgbHinge)

