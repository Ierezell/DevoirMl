# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 1
#
# ############################################################################
# ########################### INSTRUCTIONS ###################################
# ############################################################################
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
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à\
                s'exécuter! ".format(question) + "Le temps maximum permis est\
                de {0:.4f} secondes, mais votre code a requis {1:.4f}\
                secondes! ".format(maxduration, duration) + "Assurez-vous que\
                vous ne faites pas d'appels bloquants (par exemple à  show())\
                dans cette boucle!")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q1A = 1.0
TMAX_Q1B = 2.5

# Ne changez rien avant cette ligne!


# Définition de la PDF de la densité-mélange
def pdf(X):
    """Returns the mixed density of the absice array X.

    Parameters
    ----------
    X : array
        Take a column array or a matrix

    Returns
    -------
    Array
        Return an array (1, len(X)) of the density of probability

    Raises
    -------
    Array classic execptions
    """
    return 0.4 * norm(0, 1).pdf(X[:, 0]) + 0.6 * norm(5, 1).pdf(X[:, 0])


# Question 1A

# TODO Q1A
# Complétez la fonction sample(n), qui génère n
# données suivant la distribution mentionnée dans l'énoncé
def sample(n):
    """Samples n values of the mixed density (cf : pdf function).

    Parameters
    ----------
    n : int
        number of samples

    Returns
    -------
    Array of size 1, n
        List of the n samples of the mixed density
    """
    densproba = numpy.array(pdf(numpy.linspace(-5, 10, 1500).reshape(-1, 1)))
    densproba /= sum(densproba)
    return numpy.random.choice(numpy.linspace(-5, 10, 1500),
                               size=n,
                               replace=True,
                               p=densproba).reshape(-1, 1)


if __name__ == '__main__':
    # Question 1A
    _times.append(time.time())
    # TODO Q1A
    # échantillonnez 50 et 10 000 données en utilisant la fonction
    # sample(n) que vous avez définie plus haut et tracez l'histogramme
    # de cette distribution échantillonée, en utilisant 25 bins,
    # dans le domaine [-5, 10].
    # Sur les mêmes graphiques, tracez également la fonction de densité réelle.
    n = [50, 10000]
    absc50 = numpy.linspace(-5, 10, n[0])
    absc10000 = numpy.linspace(-5, 10, n[1])

    fig, (sfig1, sfig2) = pyplot.subplots(1, 2, sharex=True, sharey=True)

    sfig1.hist(sample(n[0]), bins=25,
               histtype='stepfilled', density=1, alpha=0.5)

    sfig1.plot(absc50, pdf(absc50.reshape(n[0], 1)),
               linewidth=2, alpha=0.5, color='r')

    sfig1.set_xlabel('x', fontsize=10)
    sfig1.set_ylabel('p(x)', fontsize=20)
    sfig1.set_title("50 Échantillons")

    sfig2.hist(sample(n[1]), bins=25, density=1,
               histtype='stepfilled', alpha=0.5)

    sfig2.plot(absc10000, pdf(absc10000.reshape(n[1], 1)),
               linewidth=2, alpha=0.5, color='r')

    sfig2.set_xlabel('x', fontsize=10)
    sfig2.set_ylabel('p(x)', fontsize=10)
    sfig2.set_title("10 000 Échantillons", fontsize=20)

    fig.legend(['Density', 'Hist'], loc="lower center", ncol=4, fontsize=20)
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
    # la même figure, mais tracées avec des couleurs différentes.
    fig, (sfig1, sfig2) = pyplot.subplots(1, 2, sharex=True, sharey=True)
    bandwidth = [0.3, 1, 2, 5]
    samples50 = sample(n[0]).reshape(-1, 1)
    samples10 = sample(n[1]).reshape(-1, 1)
    absc = numpy.linspace(-5, 10, 150)
    kernels = map(lambda b: KernelDensity(kernel='tophat', bandwidth=b),
                  bandwidth)

    for k in kernels:
        ker50 = numpy.exp(k.fit(samples50).score_samples(absc.reshape(-1, 1)))

        sfig1.plot(absc, ker50, alpha=0.8, lw=2)
        sfig1.scatter(absc, ker50, alpha=0.5, lw=0.5)

        ker10 = numpy.exp(k.fit(samples10).score_samples(absc.reshape(-1, 1)))

        sfig2.plot(absc, ker10, alpha=0.8, lw=2)
        sfig2.scatter(absc, ker10, alpha=0.5, lw=0.5)

    sfig1.fill(absc, pdf(absc.reshape(-1, 1)),
               linewidth=2, alpha=0.2, color='k')
    sfig1.set_ylabel('p(x)', fontsize=20)
    sfig1.set_title("50 Échantillons", fontsize=20)

    sfig2.fill(absc, pdf(absc.reshape(-1, 1)),
               linewidth=2, alpha=0.2, color='k')
    sfig2.set_ylabel('p(x)', fontsize=10)
    sfig2.set_title("10 000 Échantillons", fontsize=20)

    fig.legend(bandwidth + ['Truth'],
               loc="lower center", ncol=5, fontsize=20)

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")
    pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
