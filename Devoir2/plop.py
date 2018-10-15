# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 2
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tâche que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, à la fonction
#       checkTime, ni aux conditions vérifiant le bon fonctionnement de votre
#       code. Ces structures vous permettent de savoir rapidement si vous ne
#       respectez pas les requis minimum pour une question en particulier.
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer
#       la note de zéro (0) pour la partie implémentation!
#
###############################################################################

import time
import numpy

from matplotlib import pyplot, patches

from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Fonctions utilitaires liées à l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q2B = 5.0
TMAX_Q2Bdisp = 10.0
TMAX_Q2C = 20
TMAX_Q2Dbc = 60
TMAX_Q2Diris = 40
TMAX_Q2Ebc = 30
TMAX_Q2Eiris = 15


# Ne modifiez rien avant cette ligne!


# Question 2B
# Implémentation du discriminant linéaire
class DiscriminantLineaire:
    def __init__(self, eta=1e-2, epsilon=1e-2, max_iter=1000):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.eta = eta
        # Epsilon et max_iter servent à stocker les critères d'arrêt
        # max_iter est un simple critère considérant le nombre de mises à jour
        # effectuées sur les poids (autrement dit, on cesse l'apprentissage
        # après max_iter itération de la boucle d'entraînement), alors que
        # epsilon indique la différence minimale qu'il doit y avoir entre
        # les erreurs de deux itérations successives pour que l'on ne
        # considère pas l'algorithme comme ayant convergé. Par exemple,
        # si epsilon=1e-2, alors tant que la différence entre l'erreur
        # obtenue à la précédente itération et l'itération courante est
        # plus grande que 0.01, on continue, sinon on arrête.
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit(self, X, y):
        # Implémentez la fonction d'entraînement du classifieur, selon
        # les équations que vous avez développées dans votre rapport.

        # On initialise les poids aléatoirement
        w = numpy.random.rand(X.shape[1] + 1)

        # TODO Q2B
        # Vous devez ici implémenter l'entraînement.
        # Celui-ci devrait être contenu dans la boucle suivante, qui se répète
        # self.max_iter fois
        # Vous êtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent être contenus dans la variable w définie plus haut
        N = len(X)
        for i in range(self.max_iter):
            dwi = numpy.zeros(len(w))
            for j in range(N):
                h = numpy.vdot(w[1:], X[j]) + w[0]
                if (h * y[j] <= 0):
                    res = (y[j] - h) / pow(numpy.linalg.norm(X[j]), 2)
                    dwi[0] += res
                    dwi[1:] += X[j] * res
            w += self.eta * dwi

        # À ce stade, la variable w devrait contenir les poids entraînés
        # On les copie dans une variable membre pour les conserver
        self.w = w

    def predict(self, X):
        # TODO Q2B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        pred = numpy.zeros(X.shape[0])
        for i in range(X.shape[0]):
            pred[i] = numpy.vdot(self.w[1:], X[i]) + self.w[0]
        return pred

    def score(self, X, y):
        # TODO Q2B
        # Implémentez la fonction retournant le score (accuracy)
        # du classifieur sur les données reçues en argument.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # Indice : réutiliser votre implémentation de predict() réduit de
        # beaucoup la taille de cette fonction!
        somme = 0
        total = 0
        pred = self.predict(X)
        for i in range(len(X)):
            temp = pow(y[i] - pred[i], 2) / pow(numpy.linalg.norm(X[i]), 2)
            total += somme
            if y[i] * pred[i] <= 0:
                somme += temp
        return 1 - (somme / total)


# Question 2B
# Implémentation du classifieur un contre tous utilisant le discriminant linéaire
# défini plus haut
class ClassifieurUnContreTous:
    def __init__(self, n_classes, **kwargs):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.n_classes = n_classes
        self.estimators = [DiscriminantLineaire(
            **kwargs) for c in range(n_classes)]

    def fit(self, X, y):
        # TODO Q2C
        # Implémentez ici une approche un contre tous, où chaque classifieur
        # (contenu dans self.estimators) est entraîné à distinguer une seule classe
        # versus toutes les autres
        pass

    def predict(self, X):
        # TODO Q2C
        # Implémentez ici la prédiction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a préalablement été exécuté
        pass

    def score(self, X, y):
        # TODO Q2C
        # Implémentez ici le calcul du score utilisant l'approche un contre tous
        # Ce score correspond à la précision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        pass


if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # Problème à 2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à deux classes, et tracez les régions de décision
    y = [-1 if i == 0 else i for i in y]
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 0].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    print(xx.shape)
    disc = DiscriminantLineaire()
    disc.fit(X, y)
    err = 1 - disc.score(X, y)
    Y = disc.predict(numpy.c_[xx.ravel(), yy.ravel()])
    print((Y.reshape(xx.shape)).shape)
    fig, subfigs = pyplot.subplots(1, 1, sharex='all', sharey='all')
    subfigs.contourf(xx, yy, numpy.sign(Y.reshape(xx.shape)))
    subfigs.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")

    pyplot.show()
"""
    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à trois classes, et tracez les régions de décision

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")

    pyplot.show()

    # Question 2D

    _times.append(time.time())

    # TODO Q2D
    # Chargez les données "Breast cancer Wisconsin" et normalisez les de
    # manière à ce que leur minimum et maximum soient de 0 et 1
    cancer = load_breast_cancer()
    cancer.data = minmax_scale(cancer.data, feature_range=(0, 1))

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Breast Cancer
    # Initialisez votre discriminant linéaire avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à K=3 plis!
    discriminants = [DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000),
                     LinearDiscriminantAnalysis(),
                     Perceptron(),
                     LogisticRegression()]
    kf = KFold(n_splits=3)
    erreurs = [0., 0., 0., 0.]
    for train_index, test_index in kf.split(cancer.data, cancer.target):
        X_train, X_test = cancer.data[train_index], cancer.data[test_index]
        y_train, y_test = cancer.target[train_index], cancer.target[test_index]
        for i in range(len(discriminants)):
            discriminants[i].fit(X_train, y_train)
            erreurs[i] += discriminants[i].score(X_test, y_test)
    for i in range(len(erreurs)):
        erreurs[i] = 1 - erreurs[i] / 3

    print(erreurs)

    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")

    _times.append(time.time())
    # TODO Q2D
    # Chargez les données "Iris" et normalisez les de
    # manière à ce que leur minimum et maximum soient de 0 et 1
    iris = load_iris()
    iris.data = minmax_scale(iris.data, feature_range=(0, 1))

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Iris
    # Pour utilisez votre discriminant linéaire, utilisez l'approche Un Contre Tous
    # implémenté au 2C.
    # Initialisez vos discriminants linéaires avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à K=3 plis!

    _times.append(time.time())
    checkTime(TMAX_Q2Diris, "2Diris")

    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Breast Cancer
    # L'évaluation doit être faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de données en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (précision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et
    # scoresDistanceWeights pour weights=distance
    # Le premier élément de chacune de ces listes devrait contenir la précision
    # pour k=1, le second la précision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []

    _times.append(time.time())
    checkTime(TMAX_Q2Ebc, "2Ebc")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit être le nombre
    # de voisins et l'axe y la performance en leave-one-out

    pyplot.show()

    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Iris
    # L'évaluation doit être faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de données en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (précision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et
    # scoresDistanceWeights pour weights=distance
    # Le premier élément de chacune de ces listes devrait contenir la précision
    # pour k=1, le second la précision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []

    _times.append(time.time())
    checkTime(TMAX_Q2Eiris, "2Eiris")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit être le nombre
    # de voisins et l'axe y la performance en leave-one-out

    pyplot.show()


# N'écrivez pas de code à partir de cet endroit
"""
