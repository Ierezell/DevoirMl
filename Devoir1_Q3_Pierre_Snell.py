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
from sklearn.datasets import load_iris

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
# ########################################################################
# ############################  QUESTION 3 C #############################
# ########################################################################
class ClassifieurAvecRejet:
    """Class for homemade classifier with posibility of reject of the data

    Parameters
    ----------
    _lambda : float
        Value for reject (the default is 1).

    Examples
    -------
    >>> clf = ClassifieurAvecRejet(0.5)
    >>> clf.fit(features,target)
    >>> clf.predict(feature)
        [array of target classes]
    >>> clf.score(feature,target)
        number between 0 and 1 for positive % of good answers

    Attributes
    ----------
    _lambda
    _classes
    _counts_classes
    _Pci
    s2i
    mi

    Functions
    ----------
    fit(self, X, y)
    predict_proba(self, X)
    predict(self, X)
    score(self, X, y)

    The functions of the class raises no exeptions because I was lazy to
    implements asserts. But it should be done in predict and score to check the
    input size
    """

    def __init__(self, _lambda=1):
        # _lambda est le coût de rejet
        self._lambda = _lambda

    def fit(self, X, y):
        """Initialize the class object with estimators computed with the data.

        Parameters
        ----------
        X : list (numpy array)
            Features
        y : list (numpy array)
            Target

        Returns
        -------
        ClassifieurAvecRejet
            Class object with estimator of the data variables (mean variance)

        Raises
        -------
        nothing

        Examples
        -------

        >>> clf = ClassifieurAvecRejet(0.5)
        >>> clf.fit(features,target)

        """
        # TODO Q3C
        # Implémentez ici une fonction permettant d'entraîner votre modèle
        # à  partir des données fournies en argument
        self.X = X
        self.y = y
        # On fait une liste de classes et une de leur nombre.
        # On aurait aussi pu faire un dico avec
        # _countClasses = dict(zip(self.classes, slef._counts_classes))
        self._classes, self._counts_classes = numpy.unique(y,
                                                           return_counts=True)

        # Calcul de la probabilité à priori par compte de la classe / nb total
        # d'élements
        self._Pci = [self._counts_classes[i] / sum(self._counts_classes)
                     for i in self._classes]

        # Calcul des moyennes par _classes
        sommes = [0] * len(self._classes)
        for i in range(len(y)):
            sommes[y[i]] += X[i]
        self.mi = [sommes[i] / self._counts_classes[i] for i in self._classes]

        # Calcul de la variance par classe
        sumvar = [0] * len(self._classes)
        for i in range(len(y)):
            sumvar[y[i]] += (X[i] - self.mi[y[i]])**2
        self.s2i = [sum(sumvar[i]) / (self._counts_classes[i] * len(X[0]))
                    for i in self._classes]

        return self

    def predict_proba(self, X):
        """Predict the probability of belonging to a class.

        Parameters
        ----------
        X : list (numpy array)
            Features

        Returns
        -------
        array
            array of size len(X)*nb_class with the probability for each feature

        Raises
        -------
        nothing
        Examples
        -------
        >>> clf = ClassifieurAvecRejet(0.5)
        >>> clf.predict_proba(features)
            array[len(X)][nb_class]

        """
        # TODO Q3C
        # Implémentez une fonction retournant la probabilité d'appartenance à
        # chaque classe, pour les données passées en argument. Cette fonction
        # peut supposer que fit() a préalablement été appelé.
        # Indice : calculez les différents termes de l'équation de Bayes
        # séparément
        Proba = []
        cte = -0.5 * numpy.log(2 * numpy.pi)
        for i in self._classes:
            logVar = numpy.log(numpy.sqrt(self.s2i[i]))
            Xnorm = ((X - self.mi[i])**2) / (2 * (self.s2i[i]))
            log_Pci = numpy.log(self._Pci[i])
            logProb = cte - logVar - numpy.sum(Xnorm, axis=1) + log_Pci
            Proba.append(numpy.exp(logProb))
        Proba = numpy.transpose(Proba)

        return [Proba[i] / sum(Proba[i]) for i in range(len(Proba))]

    def predict(self, X):
        """Give the list of classes of a feature.

        Parameters
        ----------
        X : list (numpy array)
            Features

        Returns
        -------
        Array
            array with the class of the input feature of size len(X)

        Raises
        -------
        nothing

        Examples
        -------
        >>> clf = ClassifieurAvecRejet(0.5)
        >>> clf.predict(features)
            array[len(X)][nb_class]  [class1,class2..... etc...]
        >>>

        """
        # TODO Q3C
        # Implémentez une fonction retournant les prédictions pour les données
        # passées en argument. Cette fonction peut supposer que fit() a
        # préalablement été appelé.
        # Indice : vous pouvez utiliser predict_proba() pour éviter une
        # redondance du code
        alpha_i = []
        predict = self.predict_proba(X)
        for i in range(numpy.shape(X)[0]):
            if (predict[i] < 1 - self._lambda).all():
                alpha_i.append(self._classes[-1] + 1)
            else:
                alpha_i.append(numpy.argmax(predict[i]))
        return alpha_i

    def score(self, X, y):
        """return the accuracy of the classifier.

        Parameters
        ----------
            X : list (numpy array)
                Features
            y : list (numpy array)
                Target

        Returns
        -------
        float
            % of the accuracy

        Raises
        -------
        nothing

        Examples
        -------
        >>> clf = ClassifieurAvecRejet(0.5)
        >>> clf.score(features,target)
            0.8
        """
        # TODO Q3C
        # Implémentez une fonction retournant le score (tenant compte des
        # données rejetées si lambda < 1.0) pour les données passées en
        # argument.
        # Cette fonction peut supposer que fit() a préalablement été exécuté.
        return sum(self.predict(X) == y) / len(y)


if __name__ == "__main__":
    # ########################################################################
    # ############################  QUESTION 3 D #############################
    # ########################################################################
    # Question 3D

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
        sousData = data.data[:, (f1, f2)]
        # TODO Q3D
        # Créez ici une grille permettant d'afficher les régions de
        # décision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous être utile ici
        # N'utilisez pas un pas trop petit!
        pas = 100
        x1 = numpy.linspace(sousData[:, 0].min(), sousData[:, 0].max(), pas)
        x2 = numpy.linspace(sousData[:, 1].min(), sousData[:, 1].max(), pas)
        absc, ordon = numpy.meshgrid(x1, x2)
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
            clf.fit(sousData, data.target)
            # TODO Q3D
            # Obtenez et affichez son score
            # Stockez la valeur de cette erreur dans la variable err
            err = 1 - clf.score(sousData, data.target)
            # TODO Q3D
            # Utilisez la grille que vous avez créée plus haut
            # pour afficher les régions de décision, INCLUANT LA
            # ZONE DE REJET, de même que les points colorés selon
            # leur vraie classe

            # PATCH POUR LES LEGENDES......
            colors = "bgrcmykw"[:len(clf._classes) + 1]
            scat = []
            for color in colors:
                scat.append(subfig.scatter(data.data[:, f1].min(),
                                           data.data[:, f2].min(), alpha=1,
                                           c=color, s=20))

            subfig.scatter(absc, ordon, alpha=0.08, s=20,
                           c=[colors[i] for i in clf.predict(
                               numpy.c_[absc.ravel(), ordon.ravel()])])
            subfig.scatter(data.data[:, f1], data.data[:, f2], alpha=1, s=20,
                           c=[colors[i] for i in clf.predict(sousData)])
            # Identification des axes et des méthodes
            subfig.set_xlabel(str(data.feature_names[f1])[0:-5])
            subfig.set_ylabel(str(data.feature_names[f2])[0:-5])
            subfig.set_title(clf.__class__.__name__ + " " + str(clf._lambda) +
                             " Err = " + str("%.2f" % round(err, 2)),
                             fontsize=20)
            fig.legend(tuple(scat),
                       tuple(data.target_names) + tuple(["Rejet"]),
                       loc="lower center", ncol=5, fontsize=20)
        _times.append(time.time())
        checkTime(TMAX_Q3D, "3D")

        # On affiche les graphiques
        pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
