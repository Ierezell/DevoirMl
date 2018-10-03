# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 1
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
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

from matplotlib import pyplot

from scipy.stats import norm

from sklearn.neighbors import KernelDensity

# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à  s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à  show()) dans cette boucle!")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q1A = 1.0
TMAX_Q1B = 2.5

# Ne changez rien avant cette ligne!


# Définition de la PDF de la densité-mélange
def pdf(X):
    return 0.4 * norm(0, 1).pdf(X[:, 0]) + 0.6 * norm(5, 1).pdf(X[:, 0])


# Question 1A

# TODO Q1A
# Complétez la fonction sample(n), qui génère n
# données suivant la distribution mentionnée dans l'énoncé
def sample(n):
    return


if __name__ == '__main__':
    # Question 1A

    _times.append(time.time())
    # TODO Q1A
    # échantillonnez 50 et 10 000 données en utilisant la fonction
    # sample(n) que vous avez définie plus haut et tracez l'histogramme
    # de cette distribution échantillonée, en utilisant 25 bins,
    # dans le domaine [-5, 10].
    # Sur les màªmes graphiques, tracez également la fonction de densité réelle.

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1A, "1A")
    pyplot.show()

    # Question 1B
    _times.append(time.time())

    # TODO Q1B
    # échantillonnez 50 et 10 000 données, mais utilisez cette fois une
    # estimation avec noyau boxcar pour présenter les données. Pour chaque
    # nombre de données (50 et 10 000), vous devez présenter les distributions
    # estimées avec des tailles de noyau (bandwidth) de {0.3, 1, 2, 5}, dans
    # la màªme figure, mais tracées avec des couleurs différentes.

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")
    pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
