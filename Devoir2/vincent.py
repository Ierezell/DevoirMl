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
    def __init__(self, eta=2e-2, epsilon=1e-2, max_iter=1000):
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
        y=[-1 if i == 0 else i for i in y]
        # On initialise les poids aléatoirement
        w = numpy.random.rand(X.shape[1] + 1)
        self.erreur=10#initialisation plus grand que 1
        # TODO Q2B
        # Vous devez ici implémenter l'entraînement.
        # Celui-ci devrait être contenu dans la boucle suivante, qui se répète
        # self.max_iter fois
        # Vous êtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent être contenus dans la variable w définie plus haut
        for i in range(self.max_iter):
            # Nous allons tout d'abord calculer l'ensemble y qui est l'ensemble
            # données mal classés
            indices_erreurs = []
            for k in range(numpy.size(X[:, 0])):
                if(y[k] * (numpy.dot(w[1:], X[k]) + w[0]) <= 0):
                    indices_erreurs.append(k)
            erreur=len(indices_erreurs)/numpy.size(X[:,0])
            if (numpy.abs(erreur-self.erreur)<self.epsilon):
                print(i)
                break
            else:
                self.erreur=erreur
            # une fois les donnés mal-classées trouvées, nous allons calculer Delta
            # pour w et w0 par la methode du gradient
            Delta = []

            for k in range(numpy.size(w)):
                somme = 0
                for i in indices_erreurs:
                    xik = 1
                    if (k != 0):
                        xik = X[i, k-1]
                    somme += (y[i] - numpy.dot(w[1:], X[i, :]) - w[0]) * xik / numpy.dot(X[i, :], X[i, :])
                Delta.append(somme * self.eta)

            # En ajoutant Delta au vecteur w, nous recalculons le nouveau w qui
            # diminue l'erreur sur le classement. En reproduisant l'operation, nous
            # tendons vers une minimisation (globale ou locale) de l'erreur.
            w += Delta

        # À ce stade, la variable w devrait contenir les poids entraînés
        # On les copie dans une variable membre pour les conserver
        self.w = w

    def predict(self, X):
        # TODO Q2B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté

        # Nous devons mainteant calculer le resultat de la fonction disciminante
        # sur chaque ligne de X
        ycalcul = []
        for k in range(numpy.size(X[:, 0])):
            if ((numpy.dot(self.w[1:], X[k, :]) + self.w[0]) >= 0):
                ycalcul.append(1)
            else:
                ycalcul.append(0)
        return ycalcul



    def score(self, X, y):
        # TODO Q2B
        # Implémentez la fonction retournant le score (accuracy)
        # du classifieur sur les données reçues en argument.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # Indice : réutiliser votre implémentation de predict() réduit de
        # beaucoup la taille de cette fonction!
        ycalcul = self.predict(X)
        diff = ycalcul - y
        score = 1 - (numpy.size(numpy.where(diff != 0)) / numpy.size(X[:, 0]))
        return score


# Question 2B
# Implémentation du classifieur un contre tous utilisant le discriminant linéaire
# défini plus haut
class ClassifieurUnContreTous:
    def __init__(self, n_classes, **kwargs):
        # Cette fonction est déjà codée pour vous, vous n'avez qu'à utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        self.n_classes = n_classes
        self.estimators = [DiscriminantLineaire(**kwargs) for c in range(n_classes)]

    def fit(self, X, y):
        # TODO Q2C
        # Implémentez ici une approche un contre tous, où chaque classifieur
        # (contenu dans self.estimators) est entraîné à distinguer une seule classe
        # versus toutes les autres
        for k in range(self.n_classes):
            #créons le vecteur y pour chaque classées
            new_y=y-k
            new_y[numpy.where(new_y!=0)]=1
            #ainsi dans le y la classe i vaut 0 et les autres classes sont à 1

            #Nous allons maintenant entraîner chaque disciminant linéaire avec X et new_y
            self.estimators[k].fit(X,new_y)
            #Ainsi nous entraînons la classe k contre toutes les autres

    def predict(self, X):
        # TODO Q2C
        # Implémentez ici la prédiction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a préalablement été exécuté
        y_estime=[]
        for x in X:
            h_calcul=[numpy.dot(discri.w[1:], x) + discri.w[0] for discri in self.estimators]
            y_estime.append(numpy.argmax(h_calcul))#ceci ajoute à y_estime le numero de la classe qui minimise l'erreur
        return y_estime


    def score(self, X, y):
        # TODO Q2C
        # Implémentez ici le calcul du score utilisant l'approche un contre tous
        # Ce score correspond à la précision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        ycalcul = self.predict(X)
        diff = ycalcul - y
        score = 1 - (numpy.size(numpy.where(diff != 0)) / numpy.size(X[:, 0]))
        return score

if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # Problème à 2 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à deux classes, et tracez les régions de décision

    h = 0.02
    maxf1 = numpy.max(X[:, 0])
    maxf2 = numpy.max(X[:, 1])
    minf1 = numpy.min(X[:, 0])
    minf2 = numpy.min(X[:, 1])
    vectx = numpy.arange(minf1, maxf1 + h, h)
    vecty = numpy.arange(minf2, maxf2 + h, h)
    (grillex, grilley) = numpy.meshgrid(vectx, vecty)

    discri = DiscriminantLineaire()
    discri.fit(X, y)
    dataprediction = numpy.c_[grillex.reshape(-1), grilley.reshape(-1)]
    couleurs = ["r", "g"]
    yestime = discri.predict(dataprediction)
    pyplot.scatter(grillex, grilley, s=5, c=[couleurs[i] for i in yestime], alpha=0.1)
    pyplot.scatter(X[:, 0], X[:, 1], s=5, c=[couleurs[i] for i in y])
    print(discri.score(X,y))

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")

    pyplot.show()

    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    # TODO Q2C
    # Testez la performance du discriminant linéaire pour le problème
    # à trois classes, et tracez les régions de décision

    h = 0.02
    maxf1 = numpy.max(X[:, 0])
    maxf2 = numpy.max(X[:, 1])
    minf1 = numpy.min(X[:, 0])
    minf2 = numpy.min(X[:, 1])
    vectx = numpy.arange(minf1, maxf1 + h, h)
    vecty = numpy.arange(minf2, maxf2 + h, h)
    (grillex, grilley) = numpy.meshgrid(vectx, vecty)

    discri = ClassifieurUnContreTous(n_classes=3)
    discri.fit(X, y)
    dataprediction = numpy.c_[grillex.reshape(-1), grilley.reshape(-1)]
    couleurs = ["r", "g","b"]
    yestime = discri.predict(dataprediction)
    pyplot.scatter(grillex, grilley, s=5, c=[couleurs[i] for i in yestime], alpha=0.1)
    pyplot.scatter(X[:, 0], X[:, 1], s=5, c=[couleurs[i] for i in y])
    print(discri.score(X,y))


    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")

    pyplot.show()

    # Question 2D

    _times.append(time.time())

    # TODO Q2D
    # Chargez les données "Breast cancer Wisconsin" et normalisez les de
    # manière à ce que leur minimum et maximum soient de 0 et 1
    data_bc=load_breast_cancer()
    #fixons la valeur max à 1 en divisant par la valeur max
    X,y=data_bc.data/numpy.max(data_bc.data),data_bc.target

    # TODO Q2D
    # Comparez les diverses approches demandées dans l'énoncé sur Breast Cancer
    # Initialisez votre discriminant linéaire avec les paramètres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par défaut
    # N'oubliez pas que l'évaluation doit être faite par une validation
        # croisée à K=3 plis!

    #Créons la liste des classificateurs que nous allons utiliser
    Classificateurs=[DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000),LinearDiscriminantAnalysis(),Perceptron(),LogisticRegression()]

    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")

    _times.append(time.time())
    # TODO Q2D
    # Chargez les données "Iris" et normalisez les de
    # manière à ce que leur minimum et maximum soient de 0 et 1

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
