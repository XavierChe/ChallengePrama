import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pygam
from pygam import LinearGAM, s, f
import csv
import datetime

###Lecture des dossiers
df = pd.read_csv('train_data.csv')
t = pd.read_csv('test_data.csv')

###Changement de la date en nombre
D = []  # liste avec les futures dates en nombre
e = 0
for k in range(len(df.date)):
    e = df.date[k][:10]
    transition = datetime.datetime.strptime(e, '%Y-%m-%d')
    D.append(int(transition.strftime('%Y%m%d')))

D2 = []  # liste avec les dates en nombreds de l'échantillon test
e = 0
for k in range(len(t.date)):
    e = t.date[k][:10]
    transition = datetime.datetime.strptime(e, '%Y-%m-%d')
    D2.append(int(transition.strftime('%Y%m%d')))

df.insert(21, "date2", D, True)


###Transformation de colonne en log ou exp
def transfo_log(L):
    L2 = []
    for k in range(len(L)):
        L2.append(np.log(abs(L[k])))
    return (L2)


def transfo_exp(L):
    L2 = []
    for k in range(len(L)):
        L2.append(np.exp(L[k]))
    return (L2)


##log du jardin, gain 1000
jardin = transfo_log(df.m2_jardin)
df.insert(22, "log_m2_jardin", jardin, True)

j2 = transfo_log(t.m2_jardin)  # avec les différents log, score 149 000

##design en log
design = transfo_log(df.design_note)
df.insert(23, "design", design, True)

design2 = transfo_log(t.design_note)

###Creation des variables X et Y
X = df[['nb_chambres', 'nb_sdb', 'm2_interieur', 'log_m2_jardin',
        'm2_etage', 'm2_soussol', 'nb_etages', 'vue_mer', 'vue_note',
        'etat_note', 'design', 'annee_construction', 'annee_renovation',
        'm2_interieur_15voisins', 'm2_jardin_15voisins', 'lat', 'long', 'zipcode',
        'date2']]

Y = df['prix']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=111, test_size=0.2)

###Cluster
# kmeans = KMeans(n_clusters = 3, random_state = 0)
# kmeans.fit(X)


###Regression_linéaire score 190 000
# regr = linear_model.LinearRegression()


###LASSO score 190 000
# regr = Lasso(alpha=1)   #alpha = 1 donne le modèle linéaire, augmenter alpha augmente le biais


###RIDGE score 190 000
# regr = Ridge(alpha=1)


###Polynome score 186 000
# regr =  make_pipeline(PolynomialFeatures(3), Ridge(alpha=1))


###GAM score 158 000
regr = LinearGAM(s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7) + s(8) + s(9) +
                 s(10) + s(11) + s(12) + s(13) + s(14) + s(15) + s(16) + s(17))

###Bagging score 115 000
# regr = BaggingRegressor( n_estimators=10, random_state=0,verbose=1)


regr.fit(X_train, Y_train)

# Boosting score 133 000
# X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# regr = GradientBoostingRegressor(random_state=0)
# regr.fit(X_train, y_train)


###Test du RMSE sur l'échantillon test
Y_testpredict = regr.predict(X_test)
s = 0
Y_test2 = list(Y_test)
for k in range(len(Y_test2)):
    s = s + (Y_test2[k] - Y_testpredict[k]) * (Y_test2[k] - Y_testpredict[k])
s = s / len(Y_test2)
s = np.sqrt(s)
print(s)

###Retourner les prédictions
N = len(t.id)
P = []

for k in range(N):
    P.append(regr.predict([[t.nb_chambres[k], t.nb_sdb[k], t.m2_interieur[k],
                            j2[k], t.m2_etage[k], t.m2_soussol[k],
                            t.nb_etages[k], t.vue_mer[k], t.vue_note[k],
                            t.etat_note[k], design2[k], t.annee_construction[k],
                            t.annee_renovation[k], t.m2_interieur_15voisins[k],
                            t.m2_jardin_15voisins[k], t.lat[k], t.long[k], t.zipcode[k],
                            D2[k]]])[0])

ID = t.id

entetes = [u'id', u'prix']
valeurs = [ID, P]

with open('resultat2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['id', 'prix'])
    for k in range(N):
        spamwriter.writerow([ID[k], P[k]])


