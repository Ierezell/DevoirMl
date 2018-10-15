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
    def __init__(self, eta=2e-2, epsilon=1e-6, max_iter=1000):
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
        w = np.array(w)
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
            ind_mal_classe = []
            deltawi = np.zeros(len(w) - 1)
            deltaw0 = np.zeros(1)
            w = w / np.linalg.norm(w)
            for j in range(len(y)):
                hxw = np.dot(w[1:], X[j]) + w[0]
                if (hxw * y[j]) <= 0:
                    ind_mal_classe.append(j)
                    num = (y[j] - hxw) / np.linalg.norm(X[j])**2
                    deltawi += num * X[j]
                    deltaw0 += num

            normeX = np.linalg.norm(X[ind_mal_classe])**2
            numX = np.array(
                [(y[i] - np.dot(w[1:], X[i]) + w[0]) / normeX
                 for i in ind_mal_classe])

            Err = 0.5 * sum(numX**2 / normeX)
            w[0] += self.eta * deltaw0
            w[1:] += self.eta * deltawi

            # print(Err, Err_prec)

            if (Err_prec - Err) < self.epsilon:
                print("Les poids ont convergés en : ", i, " itérations")
                break
            else:
                Err_prec = Err

            """
            deltawi = self.eta * np.array(
                [np.dot(num / (normeX), X[ind_mal_classe][:, i])
                 for i in range(X.shape[1])])
            deltaw0 = self.eta * sum(num / normeX)
            """
            # à ce stade, la variable w devrait contenir les poids entraînés
            # On les copie dans une variable membre pour les conserver
        self.w = w

    def predict(self, X):
        # TODO Q2B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        return [np.sign(np.dot(self.w[1:], X[i]) + self.w[0]).astype(int)
                for i in range(len(X))]

        """
        np.array([1 if np.dot(self.w[1:], x) + self.w[0] >= 0
             else 0 for x in X])
        """

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
    def __init__(self, n_classes, method='argmax', **kwargs):
        # Cette fonction est déjà  codée pour vous, vous n'avez qu'à  utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.method = method
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
            target[np.where(target == i)] = -1
            target[np.where(target != -1)] = 1
            # target = (target*2)-1 pour mettre entre -1 et 1
            # mais déjà implémenté dans le fit de Disciminant linéraire
            self.estimators[i].fit(X, target)

    def predict(self, X):
        # TODO Q2C
        # Implémentez ici la prédiction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a préalablement été exécuté

        if self.method == 'argmax':
            classes = []
            for i in range(len(X)):
                Hall = -np.array([np.dot(est.w[1:], X[i]) + est.w[0]
                                 for est in self.estimators])

                classes.append(np.argmax(Hall))
            return classes

        elif self.method == 'posval':
            classes = []
            for i in range(len(X)):
                Hall = -np.array([np.dot(est.w[1:], X[i]) + est.w[0]
                                 for est in self.estimators])
                classes.append(np.argmax(Hall) if sum(Hall >= 0) == 1
                               else self.n_classes + 1)
            # pour faire le produit de tout les elements d'une liste
            # reduce(lambda x, y: x*y, Hall)
            # np.prod(np.array(Hall))
            return classes

    def score(self, X, y):
        # TODO Q2C
        # Implémentez ici le calcul du score utilisant l'approche un contre
        # tous. Ce score correspond à  la précision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a préalablement été exécuté

        if self.method == "posval":
            somme = 0
            for i in range(len(X)):
                pred = self.predict([X[i]])[0]
                if 0 <= pred <= self.n_classes:
                    somme += pred
            return somme / len(y)

        elif self.method == "argmax":
            return sum([self.predict(X)[i] == y[i]
                        for i in range(len(y))]) / len(y)

        return somme / len(y)


if __name__ == '__main__':
    """
    # Question 2C
    _times.append(time.time())
    # Problème à  2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)
    if set(y) == {0, 1}:
        y = (y * 2) - 1
    elif set(y) != {-1, 1}:
        raise Exception("The target must be {0,1} or {-1,1}")

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

    colors = ["bgr"[i] for i in clf.predict(
        np.c_[absc.ravel(), ordon.ravel()])]
    # sfig.contourf(absc, ordon, colors)
    sfig.scatter(absc, ordon, alpha=0.5, s=20, c=colors)
    # c=["bgrcmykw"[i]
    # for i in clf.predict(np.c_[absc.ravel(), ordon.ravel()])])

    sfig.scatter(X[:, 0], X[:, 1], c=["bgr"[i] for i in y])
    sfig.set_title("Erreur : " + str("%.2f" % round(1 - clf.score(X, y), 2)),
                   fontsize=20)

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")
    plt.show()
    """
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
    clf = ClassifieurUnContreTous(
        n_classes=3, method='argmax', eta=2e-2, epsilon=1e-6)
    clf.fit(X, y)
    print("Score 3 classes : ", clf.score(X, y))

    fig, sfig = plt.subplots(1, 1, sharex=True, sharey=True)
    colors = ["bgrcmykw"[i] for i in clf.predict(
        np.c_[absc.ravel(), ordon.ravel()])]
    sfig.scatter(absc, ordon, alpha=0.5, s=20,
                 c=colors)
    sfig.scatter(X[:, 0], X[:, 1], c=["bgrcmykw"[i] for i in y])

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")

    plt.show()

    # Question 2D
    """
    _times.append(time.time())

    # TODO Q2D
    # Chargez les données "Breast cancer Wisconsin" et normalisez les de
    # manière à  ce que leur minimum et maximum soient de 0 et 1
    data = load_breast_cancer()
    X, y = minmax_scale(data.data, feature_range=(0, 1)), data.target

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Breast Cancer
    # Initialisez votre discriminant linéaire avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à  K=3 plis!
    clfs = [DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=100),
            LinearDiscriminantAnalysis(),
            Perceptron(),
            LogisticRegression()]

    kf = KFold(n_splits=3, random_state=666, shuffle=False)
    ErrorsTest = [0] * 4
    ErrorsTrain = [0] * 4
    for i in range(len(clfs)):
        avgErrorTest = 0
        avgErrorTrain = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clfs[i].fit(X_train, y_train)
            avgErrorTrain += 1 - clfs[i].score(X_train, y_train)
            avgErrorTest += 1 - clfs[i].score(X_test, y_test)
        ErrorsTest[i] = avgErrorTest / 3
        ErrorsTrain[i] = avgErrorTrain / 3

    print("Erreurs cancer Train (Own, Linear, Perceptron, Logistic) :\n",
          ErrorsTrain)
    print("Erreurs cancer Test (Own, Linear, Perceptron, Logistic) :\n",
          ErrorsTest)

    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")

    _times.append(time.time())
    # TODO Q2D
    # Chargez les données "Iris" et normalisez les de
    # manière à  ce que leur minimum et maximum soient de 0 et 1
    data = load_iris()
    X, y = minmax_scale(data.data, feature_range=(0, 1)), data.target

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Iris
    # Pour utilisez votre discriminant linéaire, utilisez l'approche Un Contre
    # Tous implémenté au 2C.
    # Initialisez vos discriminants linéaires avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
    # croisée à  K=3 plis!

    clfs = [ClassifieurUnContreTous(len(X[0]), eta=1e-4,
                                    epsilon=1e-6, max_iter=10000),
            LinearDiscriminantAnalysis(),
            Perceptron(),
            LogisticRegression()]

    kf = KFold(n_splits=3, random_state=666, shuffle=True)
    ErrorsTest = [0] * 4
    ErrorsTrain = [0] * 4
    for i in range(len(clfs)):
        avgErrorTest = 0
        avgErrorTrain = 0
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clfs[i].fit(X_train, y_train)

            avgErrorTrain += 1 - clfs[i].score(X_train, y_train)
            avgErrorTest += 1 - clfs[i].score(X_test, y_test)
        ErrorsTest[i] = avgErrorTest / 3
        ErrorsTrain[i] = avgErrorTrain / 3

    print("Erreurs Iris Train (Own, Linear, Perceptron, Logistic) :\n",
          ErrorsTrain)
    print("Erreurs Iris Test (Own, Linear, Perceptron, Logistic) :\n",
          ErrorsTest)

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
    data = load_breast_cancer()
    X, y = minmax_scale(data.data, feature_range=(0, 1)), data.target

    K = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
    scoresUniformWeights = []
    scoresDistanceWeights = []
    avgErrUni = 0
    avgErrDist = 0
    loo = LeaveOneOut()

    KnnUni = map(lambda k: KNeighborsClassifier(n_neighbors=k,
                                                weights='uniform'), K)
    KnnDist = map(lambda k: KNeighborsClassifier(n_neighbors=k,
                                                 weights='distance'), K)

    for Kuni, Kdist in zip(list(KnnUni), list(KnnDist)):
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            Kuni.fit(X_train, y_train)
            Kdist.fit(X_train, y_train)

            avgErrUni += Kuni.score(X_test, y_test)
            avgErrDist += Kdist.score(X_test, y_test)
        scoresUniformWeights.append(avgErrUni / loo.get_n_splits(X))
        scoresDistanceWeights.append(avgErrDist / loo.get_n_splits(X))

    _times.append(time.time())
    checkTime(TMAX_Q2Ebc, "2Ebc")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit être
    # le nombre de voisins et l'axe y la performance en leave-one-out
    fig, sfig = plt.subplots(1, 1, sharex=True, sharey=True)
    sfig.scatter(K, scoresUniformWeights, alpha=0.8, s=10)
    sfig.plot(K, scoresUniformWeights)
    sfig.scatter(K, scoresDistanceWeights, alpha=0.8, s=10)
    sfig.plot(K, scoresDistanceWeights)
    sfig.set_xlabel("K")
    sfig.set_ylabel("Score")
    fig.legend(("Uniform", "Distance"),
               loc="lower center", ncol=2, fontsize=20)

    plt.show()

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
    data = load_iris()
    X, y = minmax_scale(data.data, feature_range=(0, 1)), data.target

    K = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
    scoresUniformWeights = []
    scoresDistanceWeights = []
    avgErrUni = 0
    avgErrDist = 0
    loo = LeaveOneOut()

    KnnUni = map(lambda k: KNeighborsClassifier(n_neighbors=k,
                                                weights='uniform'), K)
    KnnDist = map(lambda k: KNeighborsClassifier(n_neighbors=k,
                                                 weights='distance'), K)

    for Kuni, Kdist in zip(list(KnnUni), list(KnnDist)):
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            Kuni.fit(X_train, y_train)
            Kdist.fit(X_train, y_train)

            avgErrUni += Kuni.score(X_test, y_test)
            avgErrDist += Kdist.score(X_test, y_test)
        scoresUniformWeights.append(avgErrUni / loo.get_n_splits(X))
        scoresDistanceWeights.append(avgErrDist / loo.get_n_splits(X))

    _times.append(time.time())
    checkTime(TMAX_Q2Eiris, "2Eiris")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit être le
    # nombre de voisins et l'axe y la performance en leave-one-out
    fig, sfig = plt.subplots(1, 1, sharex=True, sharey=True)
    sfig.scatter(K, scoresUniformWeights, alpha=0.8, s=5)
    sfig.plot(K, scoresUniformWeights)
    sfig.scatter(K, scoresDistanceWeights, alpha=0.8, s=5)
    sfig.plot(K, scoresDistanceWeights)
    sfig.set_xlabel("K", fontsize=20)
    sfig.set_ylabel("Score", fontsize=20)
    fig.legend(("Uniform", "Distance"),
               loc="lower center", ncol=2, fontsize=20)
    plt.show()

    # N'écrivez pas de code à  partir de cet endroit
    """
