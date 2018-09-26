#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# Introduction à  l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 3
#
###############################################################################
#                            INSTRUCTIONS                                     #
###############################################################################
#
# - Repérez les commentaires commenà§ant par TODO : ils indiquent une tà¢che
#   que vous devez effectuer.
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

from matplotlib import pyplot
from sklearn.datasets import make_moons, load_iris

# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à \
                s'exécuter! ".format(question) + "Le temps maximum permis est \
                de {0:.4f} secondes, mais votre code a requis {1:.4f}\
                secondes!".format(maxduration, duration) + "Assurez-vous que \
                vous ne faites pas d'appels bloquants (par exemple à  show())\
                dans cette boucle!")


TMAX_Q3D = 1.0


# Ne modifiez rien avant cette ligne!


# Question 3C
class ClassifieurAvecRejet:

    def __init__(self, _lambda=1):
        # _lambda est le coût de rejet
        self._lambda = _lambda

    def fit(self, X, y):
        # TODO Q3C
        # Implémentez ici une fonction permettant d'entraîner votre modèle
        # à  partir des données fournies en argument
        unique, counts = numpy.unique(y, return_counts=True)
        countClass = dict(zip(unique, counts))
        self.Pci = [countClass[i] / sum(countClass.values())
                    for i in countClass.keys()]
        # [counts[i] / sum(counts) for i in range(len(counts))]

        sommes = [0] * len(set(y))
        for i in range(len(y)):
            sommes[y[i]] += X[i]
        self.mi = [sommes[i] / countClass[i] for i in countClass.keys()]

        sumvar = [0] * len(countClass.keys())
        for i in range(len(y)):
            sumvar[y[i]] += (X[i] - mi[y[i]])**2
        self.s2i = [sumvar[i] / countClass[i] for i in countClass.keys()]

        self.pxCi = [0] * len(countClass.keys())
        for i in range(len(y)):
            self.pxCi[y[i]] = 1 / numpy.sqrt(2 * numpy.pi * s2i[y[i]]) * \
                numpy.exp(-((X[i] - mi[y[i]])**2) / (2 * s2i[y[i]]))
        self.X = X
        self.y = y
        return self

    def predict_proba(self, X):
        # TODO Q3C
        # Implémentez une fonction retournant la probabilité d'appartenance à
        # chaque classe, pour les données passées en argument. Cette fonction
        # peut supposer que fit() a préalablement été appelé.
        # Indice : calculez les différents termes de l'équation de Bayes
        # séparément
        return numpy.argmax(self.pxCi * self.Pci)

    def predict(self, X):
        # TODO Q3C
        # Implémentez une fonction retournant les prédictions pour les données
        # passées en argument. Cette fonction peut supposer que fit() a
        # préalablement été appelé.
        # Indice : vous pouvez utiliser predict_proba() pour éviter une
        # redondance du code
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y):
        # TODO Q3C
        # Implémentez une fonction retournant le score (tenant compte des
        # données rejetées si lambda < 1.0) pour les données passées en
        # argument.
        # Cette fonction peut supposer que fit() a préalablement été exécuté.


        # Question 3D
if __name__ == "__main__":

    # TODO Q3D
    # Chargez ici le dataset 'iris' dans une variable nommée data
    data = load_iris()

    # Cette ligne crée une liste contenant toutes les paires
    # possibles entre les 4 mesures
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    # Utilisons cette liste de paires pour tester le classifieur
    # avec différents lambda
    for (f1, f2) in pairs:
        # TODO Q3D
        # Créez ici un sous-dataset contenant seulement les
        # mesures désignées par f1 et f2

        # TODO Q3D
        # Créez ici une grille permettant d'afficher les régions de
        # décision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous àªtre utile ici
        # N'utilisez pas un pas trop petit!

        # On initialise les classifieurs avec différents paramètres lambda
        classifieurs = [ClassifieurAvecRejet(0.1),
                        ClassifieurAvecRejet(0.3),
                        ClassifieurAvecRejet(0.5),
                        ClassifieurAvecRejet(1)]

        # On crée une figure à  plusieurs sous-graphes pour pouvoir montrer,
        # pour chaque configuration, les régions de décisions, incluant
        # la zone de rejet
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())
        for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):

            # TODO Q3D
            # Entraînez le classifieur

            # TODO Q3D
            # Obtenez et affichez son score
            # Stockez la valeur de cette erreur dans la variable err

            # TODO Q3D
            # Utilisez la grille que vous avez créée plus haut
            # pour afficher les régions de décision, INCLUANT LA
            # ZONE DE REJET, de màªme que les points colorés selon
            # leur vraie classe

            # On ajoute un titre et des étiquettes d'axes
            subfig.set_title("lambda=" + str(clf._lambda))
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
        _times.append(time.time())
        checkTime(TMAX_Q3D, "3D")

        # On affiche les graphiques
        pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
