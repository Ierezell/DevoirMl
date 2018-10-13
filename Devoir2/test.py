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
import numpy as np

from matplotlib import pyplot as plt
# from matplotlib import patches

from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.preprocessing import minmax_scale
# from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à\
                s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre\
              code a requis {1:.4f} secondes! ".format(maxduration, duration) +
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
    def __init__(self, eta=5e-2, epsilon=1e-3, max_iter=1000):
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
        w = np.random.rand(X.shape[1] + 1)
        # Je prend w[0] = w0 et w[1:] tout le reste des poids
        # TODO Q2B
        # Vous devez ici implémenter l'entraînement.
        # Celui-ci devrait être contenu dans la boucle suivante, qui se répète
        # self.max_iter fois
        # Vous êtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent être contenus dans la variable w définie
        # plus haut
        Err_prec = 1e10
        for i in range(self.max_iter):
            # print("boucle : ", i)
            num_mal_classe = []
            ind_mal_classe = []

            for j in range(len(y)):
                hxjw = np.dot(w[1:], X[j]) + w[0]
                Y = y[j] * hxjw
                if Y <= 0:
                    num_mal_classe.append(y[j] - hxjw)
                    ind_mal_classe.append(j)
            ind_mal_classe = np.array(ind_mal_classe)
            if ind_mal_classe.size:
                normeX = np.linalg.norm(X[ind_mal_classe])**2
                num = np.array(num_mal_classe)
                Err = np.array(0.5 * sum(num**2 / normeX))
                deltawi = self.eta * np.array(
                    [np.dot(num / (normeX), X[ind_mal_classe][:, i])
                     for i in range(X.shape[1])])
                deltaw0 = self.eta * sum(num / normeX)
            else:
                Err = 0
                deltawi = 0
                deltaw0 = 0

            # condition d'arrêt :
            # print("Err : ", Err)
            # print()
            if Err_prec - Err < self.epsilon:
                print("Les poids ont convergés")
                break
            else:
                Err_prec = Err
            w[1:] += deltawi
            w[0] += deltaw0

            # à ce stade, la variable w devrait contenir les poids entraînés
            # On les copie dans une variable membre pour les conserver
        self.w = w

    def predict(self, X):
        # TODO Q2B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        return np.array(
            [1 if np.dot(self.w[1:], x) + self.w[0] >= 0
             else 0 for x in X])

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
            target = y.copy()
            target[np.where(target == i)] = 0
            target[np.where(target != 0)] = 1
            # target = (target*2)-1 pour mettre entre -1 et 1
            # mais déjà implémenté dans le fit de Disciminant linéraire
            self.estimators[i].fit(X, target)

    def predict(self, X):
        # TODO Q2C
        # Implémentez ici la prédiction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a préalablement été exécuté
        classes = []
        for x in X:
            Hall = [np.dot(est.w[1:], x) + est.w[0] for est in self.estimators]
            classes.append(np.argmax(Hall))
        return classes

    def score(self, X, y):
        # TODO Q2C
        # Implémentez ici le calcul du score utilisant l'approche un contre
        # tous. Ce score correspond à  la précision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        return sum([self.predict(X)[i] == y[i]
                    for i in range(len(y))]) / len(y)


if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # Problème à  2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à  deux classes, et tracez les régions de décision

    pas = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), pas)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), pas)
    absc, ordon = np.meshgrid(x1, x2)

    clf = DiscriminantLineaire()
    clf.fit(X, y)
    print("Score 2 Classes: ", clf.score(X, y))
    fig, sfig = plt.subplots(1, 1, sharex=True, sharey=True)
    """
    sfig.plot(np.array(plt.gca().get_xlim(),
                          clf.w[0] + clf.w[1:] *
                          np.array(plt.gca().get_xlim()),
                          'ro-'))
    """
    sfig.scatter(absc, ordon, alpha=0.5, s=20,
                 c=["bgrcmykw"[i] for i in clf.predict(
                     np.c_[absc.ravel(), ordon.ravel()])])
    sfig.scatter(X[:, 0], X[:, 1], c=["bgrcmykw"[i] for i in y])
    sfig.plot()

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")
    plt.show()
    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à  trois classes, et tracez les régions de décision
    pas = 100
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), pas)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), pas)
    absc, ordon = np.meshgrid(x1, x2)
    clf = ClassifieurUnContreTous(n_classes=3)
    clf.fit(X, y)
    print("Score 3 classes : ", clf.score(X, y))

    fig, sfig = plt.subplots(1, 1, sharex=True, sharey=True)

    sfig.scatter(absc, ordon, alpha=0.5, s=20,
                 c=["bgrcmykw"[i] for i in clf.predict(
                     np.c_[absc.ravel(), ordon.ravel()])])
    sfig.scatter(X[:, 0], X[:, 1], c=["bgrcmykw"[i] for i in y])

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")

    plt.show()
