# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 2
#
# #############################################################################
# ############################ INSTRUCTIONS ###################################
# #############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tâche que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, à  la fonction
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

# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à\
                s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code\
              a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants \
              (par exemple à  show()) dans cette boucle!")


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
        # Cette fonction est déjà  codée pour vous, vous n'avez qu'à  utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.eta = eta
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit(self, X, y):
        if set(y) == {0, 1}:
            y = (y * 2) - 1
        elif set(y) != {-1, 1}:
            raise Exception("The target must be {0,1} or {-1,1}")
        # Implémentez la fonction d'entraînement du classifieur, selon
        # les équations que vous avez développées dans votre rapport.

        # On initialise les poids aléatoirement
        w = numpy.random.rand(X.shape[1] + 1)

        # TODO Q2B
        # Vous devez ici implémenter l'entraînement.
        # Celui-ci devrait être contenu dans la boucle suivante, qui se répète
        # self.max_iter fois
        # Vous êtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent être contenus dans la variable w définie
        # plus haut
        Err_prev = 1e10
        for i in range(self.max_iter):
            print("boucle : ", i)
            hxw_mal_classe = []
            ind_mal_classe = []

            for j in range(len(y)):
                Y = y[j] * ((w[1:].dot(X[j])) + w[0])
                if Y <= 0:
                    hxw_mal_classe.append((w[1:].dot(X[j])) + w[0])
                    ind_mal_classe.append(j)
            ind_mal_classe = numpy.array(ind_mal_classe)
            hxw_mal_classe = numpy.array(hxw_mal_classe)

            if ind_mal_classe.size:
                normeX = numpy.linalg.norm(X[ind_mal_classe])
                num = y[ind_mal_classe] - hxw_mal_classe
                frac = (num) / (normeX**2)
                Err = 0.5 * sum((num / normeX)**2)
                deltawi = self.eta * sum(frac.dot(X[ind_mal_classe]))
                deltaw0 = self.eta * sum(frac)
            else:
                Err = 0
                deltawi = 0
                deltaw0 = 0

            # condition d'arrêt :
            print(Err)
            if Err_prev - Err < self.epsilon:
                print("Wololo !")
                break
            else:
                Err_prev = Err

            w[1:] += deltawi
            w[0] += deltaw0

            # à ce stade, la variable w devrait contenir les poids entraînés
            # On les copie dans une variable membre pour les conserver
        self.w = w

    def predict(self, X):
        # TODO Q2B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        return numpy.array(
            [1 if self.w[1:].dot(x) + self.w[0] >= 0
             else 0
             for x in X])

    def score(self, X, y):
        if set(y) == {0, 1}:
            y = (y * 2) - 1
        elif set(y) != {-1, 1}:
            raise Exception("The target must be {0,1} or {-1,1}")
        # TODO Q2B
        # Implémentez la fonction retournant le score (précision / accuracy)
        # du classifieur sur les données reçues en argument.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # Indice : réutiliser votre implémentation de predict() réduit de
        # beaucoup la taille de cette fonction!

        # Question 2B
        # Implémentation du classifieur un contre tous utilisant le
        # discriminant linéaire défini plus haut
        return sum([self.predict(X)[i] == y[i]
                    for i in range(len(y))]) / len(y)


class ClassifieurUnContreTous:
    def __init__(self, n_classes, **kwargs):
        # Cette fonction est déjà  codée pour vous, vous n'avez qu'à  utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.n_classes = n_classes
        self.estimators = [DiscriminantLineaire(
            **kwargs) for c in range(n_classes)]

    def fit(self, X, y):
        # TODO Q2C
        # Implémentez ici une approche un contre tous, oà¹ chaque classifieur
        # (contenu dans self.estimators) est entraîné à  distinguer une seule
        # classe versus toutes les autres
        for i in range(self.n_classes):
            target = y - i
            target[numpy.where(target != 0)] = 1
            # target = (target*2)-1 pour mettre entre -1 et 1
            # mais déjà implémenté dans le fit de Disciminant linéraire
            self.estimators[i].fit(X, target)

    def predict(self, X):
        # TODO Q2C
        # Implémentez ici la prédiction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a préalablement été exécuté
        classes = []
        for x in X:
            Hall = [est.w[1:].dot(x) + est.w[0] for est in self.estimators]
            classes.append(numpy.argmax(Hall))
        return classes

    def score(self, X, y):
        # TODO Q2C
        # Implémentez ici le calcul du score utilisant l'approche un contre
        # tous. Ce score correspond à  la précision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        return [self.predict(X)[i] == y[i] for i in range(len(y))]


if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # Problème à  2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à  deux classes, et tracez les régions de décision
    clf = DiscriminantLineaire()
    clf.fit(X, y)
    fig, sfig = pyplot.subplots(1, 1, sharex=True, sharey=True)
    """
    sfig.plot(numpy.array(pyplot.gca().get_xlim(),
                          clf.w[0] + clf.w[1:] *
                          numpy.array(pyplot.gca().get_xlim()),
                          'ro-'))
    """
    pas = 100
    x1 = numpy.linspace(X[:, 0].min(), X[:, 0].max(), pas)
    x2 = numpy.linspace(X[:, 1].min(), X[:, 1].max(), pas)
    absc, ordon = numpy.meshgrid(x1, x2)
    sfig.scatter(absc, ordon, alpha=0.05, s=20,
                 c=["bgrcmykw"[i] for i in clf.predict(
                     numpy.c_[absc.ravel(), ordon.ravel()])])
    sfig.scatter(X[:, 0], X[:, 1], c=["bgrcmykw"[i] for i in y])
    sfig.plot()

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")

    pyplot.show()

    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à  trois classes, et tracez les régions de décision

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")

    pyplot.show()

    # Question 2D

    _times.append(time.time())

    # TODO Q2D
    # Chargez les données "Breast cancer Wisconsin" et normalisez les de
    # manière à  ce que leur minimum et maximum soient de 0 et 1

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Breast Cancer
    # Initialisez votre discriminant linéaire avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à  K=3 plis!

    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")

    _times.append(time.time())
    # TODO Q2D
    # Chargez les données "Iris" et normalisez les de
    # manière à  ce que leur minimum et maximum soient de 0 et 1

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Iris
    # Pour utilisez votre discriminant linéaire, utilisez l'approche Un Contre
    # Tous implémenté au 2C.
    # Initialisez vos discriminants linéaires avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à  K=3 plis!

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
    # Stockez les performances obtenues
    # (précision moyenne pour chaque valeur de k)
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
    # et l'autre pour weights=distance. L'axe x de la figure doit être
    # le nombre de voisins et l'axe y la performance en leave-one-out

    pyplot.show()

    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Iris
    # L'évaluation doit être faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de données en utilisant minmax_scale!
    #
    # Stockez les performances obtenues
    # (précision moyenne pour chaque valeur de k)
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
    # et l'autre pour weights=distance. L'axe x de la figure doit être le
    # nombre de voisins et l'axe y la performance en leave-one-out

    pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
