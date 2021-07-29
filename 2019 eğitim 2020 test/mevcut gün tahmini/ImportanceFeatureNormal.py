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

allData = allData.iloc[:,1:]
dfData = allData
litre = dfData.iloc[:, 1:2].values
y = dfData.iloc[:, 0:1].values
x = dfData.iloc[:, 2:].values
x = pd.DataFrame(x)
litre = pd.DataFrame(litre)
litreShift = litre.shift(periods = 1)
litreShift = litreShift.fillna(0)
y = pd.DataFrame(y)
yShift = y.shift(periods=-1)
yShift = yShift.fillna(0)
y = round(y)


#xtrain 364 olacak
#xtest 365 olacak
#y train 364 olacak
#ytest 365 olacak

startPoint = 0
splint = 365    
#731 2021' e kadar
x_train = x.iloc[0:splint, :].values
x_test = x.iloc[splint:, :].values
y_train = y.iloc[0:splint, :].values
y_test = y.iloc[splint:, :].values


dfXTest = pd.DataFrame(x_test)
litreXTest = dfXTest.iloc[:,4:5]


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(x_train)
X_test_sc = sc.transform(x_test) #fit_transform: x_trainden ogren ve transform et, tek transform ogrenmeden kullan demek


def KnnImportanceX(x,y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.inspection import permutation_importance
    model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski')
    model.fit(x, y)
    results = permutation_importance(model, x, y, scoring='accuracy')
    importanceFeature = results.importances_mean
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('KNN')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX

'''
def KnnPred (x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xTrainSc = sc.fit_transform(xTrain)
    xTestSc = sc.transform(xTest) #fit_transform: x_trainden ogren ve transform et, tek transform ogrenmeden kullan demek
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.inspection import permutation_importance
    model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski') #n_neighbors komsu sayisi, metric, mesafe olcumu
    model.fit(xTrainSc, yTrain) #fit edemiyor, y' dekis ayılar float olduğu için fit edemediği farkeidlid ve sayılar yuvarlandı
    yPred = model.predict(xTestSc)
    model.fit(x, y)
    results = permutation_importance(model, x, y, scoring='accuracy')
    importanceFeature = results.importances_mean
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('Knn')
    pyplot.show()
    return yPred
yPredKnn = KnnPred(x,y)
yPredKnnImportance = KnnPred(xForKnn,y)
'''

xForKnn = KnnImportanceX(x,y)
xForKnn.to_excel("xForKnnYeni.xlsx") 


def dtrImportanceX(x,y):
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=0)
    importance = model.fit(x,y)  
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    dtrRegFeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('DTR')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            dtrRegFeatureIndex.append(importanceFeature.index(i))
    xForDtrReg = x[dtrRegFeatureIndex]
    return xForDtrReg


#corrDtrImpFeature = np.corrcoef(xForDtrReg)

'''
def DtrRegPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=0)
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x,y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('DecisionTreeRegressor')
    pyplot.show()
    return yPred
yPredDtr = DtrRegPred(x,y)
yPredImportanceDtr = DtrRegPred(xForDtrReg,y)
'''

xForDtrReg = dtrImportanceX(x,y)
xForDtrReg.to_excel("xForDtrRegYeni.xlsx") 

def dtrClassImportanceX(x,y):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)
    importance = model.fit(x,y)  
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('DTC')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX
'''
def DtrClassPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=0)
    model.fit(xTrain,yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x,y)
    importanceDtr = importance.feature_importances_
    importanceDtr = list(importanceDtr)
    pyplot.bar([x for x in range(len(importanceDtr))], importanceDtr)
    pyplot.title('DecisionTreeClassifier')
    pyplot.show()
    return yPred
yPredDtrClass = DtrClassPred(x,y)
yPredImportanceDtrClass = DtrClassPred(xForDtrClass,y)
'''

xForDtrClass = dtrClassImportanceX(x,y)
xForDtrClass.to_excel("xForDtrClassYeni.xlsx") 

def logisticRegressionImportanceX(x,y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    importance = model.fit(x, y)
    importanceFeature = importance.coef_[0]
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('LR')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX

'''
def LogisticRegressionPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x, y)
    importanceFeature = importance.coef_[0]
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('LogisticRegression')
    pyplot.show()
    return yPred
yPredLogisticRegression = LogisticRegressionPred(x,y)
yPredImportanceLogisticRegression = LogisticRegressionPred(xForLogisticRegression,y)

'''
xForLogisticRegression = logisticRegressionImportanceX(x,y)
xForLogisticRegression.to_excel("xForLogisticRegressionYeni.xlsx") 

def SvmImportanceX(x,y):
    from sklearn import svm
    model = svm.LinearSVC()
    importance = model.fit(x, y)
    importanceFeature = importance.coef_[0]
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('SVM')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX

'''
def SvmPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn import svm
    model = svm.LinearSVC()
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x, y)
    importanceFeature = importance.coef_[0]
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('Support Vector Machine')
    pyplot.show()
    return yPred
yPredSvm = SvmPred(x,y)
yPredImportanceSvm = SvmPred(xForSvm,y)

'''
xForSvm = SvmImportanceX(x,y)
xForSvm.to_excel("xForSvmYeni.xlsx") 


def RfClassImportanceX(x,y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    importance = model.fit(x,y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('RFC')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX
'''
def RfclassPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x, y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('RandomForestClassifier')
    pyplot.show()
    return yPred
yPredRfc = RfclassPred(x,y)
yPredImportanceRfc = RfclassPred(xForRfc,y)

'''
xForRfc = RfClassImportanceX(x,y)
xForRfc.to_excel("xForRfcYeni.xlsx") 

def NntanceX(x,y):
    #from sklearn.neural_network import MLPClassifier
    #model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    from sklearn.linear_model import Perceptron
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(x, y)
    importance = clf.coef_[0]
    importanceFeature = list(importance)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('NN')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX
'''
def NnPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    from sklearn.linear_model import Perceptron
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(x, y)
    importanceNn = clf.coef_[0]
    pyplot.bar([x for x in range(len(importanceNn))], importanceNn)
    pyplot.title('Neural Network')
    pyplot.show()
    return yPred
yPredNn = NnPred(x,y)
yPredImportanceNn = NnPred(xForNn,y)

'''
xForNn = NntanceX(x,y)
xForNn.to_excel("xForNnYeni.xlsx") 

def XgbImportanceX(x,y):
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="binary:logistic")
    importance = model.fit(x,y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    FeatureIndex = []
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('XGB')
    pyplot.show()
    for i in importanceFeature:
        if i > 0.0:
            FeatureIndex.append(importanceFeature.index(i))
    newImportanceX = x[FeatureIndex]
    return newImportanceX
'''
def XgbPred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="binary:logistic")
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x, y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('Xgb')
    pyplot.show()
    return yPred

def XgbHingePred(x,y):
    xTrain = x.iloc[0:365, :].values
    xTest = x.iloc[365:, :].values
    yTrain = y.iloc[0:365, :].values
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="binary:hinge")
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    importance = model.fit(x, y)
    importanceFeature = importance.feature_importances_
    importanceFeature = list(importanceFeature)
    pyplot.bar([x for x in range(len(importanceFeature))], importanceFeature)
    pyplot.title('Xgb')
    pyplot.show()
    return yPred
yPredXgb = XgbPred(x,y)
yPredImportanceXgb = XgbPred(xForXgb,y)
yPredXgbHinge = XgbHingePred(x,y)
yPredImportanceXgbHinge = XgbHingePred(xForXgb,y)

'''
xForXgb = XgbImportanceX(x,y)
xForXgb.to_excel("xForXgbYeni.xlsx") 

'''
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
    accScores.append('%s, Acc Score: %.5f' % (models[i],(accuracy_score(y_test, predModels[i]))))
    r2Scores.append('%s, R2 Score: %.5f' % (models[i],(r2_score(y_test, predModels[i]))))
    mseScores.append('%s, Mse Score: %.5f' % (models[i],(mean_squared_error(y_test, predModels[i]))))
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

'''