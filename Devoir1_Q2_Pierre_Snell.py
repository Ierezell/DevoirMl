###############################################################################
# Introduction à  l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 2
#
###############################################################################
#                            INSTRUCTIONS                                     #
###############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tà¢che
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

# Jeu de données utilisés
from sklearn.datasets import load_iris, make_circles

# Classifieurs utilisés
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

# Méthodes d'évaluation
from sklearn.model_selection import train_test_split, RepeatedKFold

# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0}\
                met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code\
               a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants \
              (par exemple à  show()) dans cette boucle!")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q2A = 0.5
TMAX_Q2B = 1.5
TMAX_Q2Cii = 0.5
TMAX_Q2Ciii = 0.5
TMAX_Q2D = 1.0

# Définition des erreurs maximales attendues pour chaque sous-question
ERRMAX_Q2B = 0.22
ERRMAX_Q2Cii = 0.07
ERRMAX_Q2Ciii = 0.07


# Ne changez rien avant cette ligne!
# Seul le code suivant le "if __name__ == '__main__':" comporte des sections à
#  implémenter

if __name__ == '__main__':

    # ########################################################################
    # ############################  QUESTION 2 A #############################
    # ########################################################################
    # Question 2A

    # TODO Q2A
    # Chargez ici le dataset 'iris' dans une variable nommée data
    data = load_iris()
    # Cette ligne crée une liste contenant toutes les paires
    # possibles entre les 4 mesures
    # Par exemple : [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    # Utilisons cette liste de paires pour afficher les données,
    # deux mesures à la fois
    # On crée une figure à  plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(2, 3)
    _times.append(time.time())
    for (f1, f2), subfig in zip(pairs, subfigs.reshape(-1)):
        # TODO Q2A
        # Affichez les données en utilisant f1 et f2 comme mesures
        subfig.scatter(data.data[:, f1], data.data[:, f2], alpha=0.8, s=20)
        subfig.set_xlabel(str(data.feature_names[f1])[0:-5], fontsize=20)
        subfig.set_ylabel(str(data.feature_names[f2])[0:-5], fontsize=20)
        # Affichage en enlevant (cm) pour les noms et les écrire plus gros
        # et modifiant la taille et la transparence des points
    _times.append(time.time())
    checkTime(TMAX_Q2A, "2A")

    # On affiche la figure
    pyplot.show()

    # ########################################################################
    # ############################  QUESTION 2 B #############################
    # ########################################################################
    # Question 2B

    # Reprenons les paires de mesures, mais entraînons cette fois
    # différents modèles demandés avec chaque paire
    for (f1, f2) in pairs:
        # TODO Q2B
        # Créez ici un sous-dataset contenant seulement les
        # mesures désignées par f1 et f2
        sousData = data.data[:, (f1, f2)]
        # TODO Q2B
        # Initialisez ici les différents classifieurs, dans
        # une liste nommée "classifieurs"
        classifieurs = [QuadraticDiscriminantAnalysis(),
                        LinearDiscriminantAnalysis(),
                        GaussianNB(),
                        NearestCentroid()
                        ]

        # TODO Q2B
        # Créez ici une grille permettant d'afficher les régions de
        # décision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous être utile ici
        # N'utilisez pas un pas trop petit!
        pas = 100
        x = numpy.linspace(sousData[:, 0].min(), sousData[:, 0].max(), pas)
        y = numpy.linspace(sousData[:, 1].min(), sousData[:, 1].max(), pas)
        absc, ordon = numpy.meshgrid(x, y)

        # On crée une figure à  plusieurs sous-graphes
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())

        # On crée un tableau du nombre de classes de couleurs
        nb_class = len(set(data.target))
        colors = "bgrcmykw"[:nb_class]
        for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):
            # TODO Q2B
            # Entraînez le classifieur
            clf.fit(sousData, data.target)
            # TODO Q2B
            # Obtenez et affichez son erreur (1 - accuracy)
            # Stockez la valeur de cette erreur dans la variable err
            err = 1 - clf.score(sousData, data.target)
            # Erreur sur le set d'entraienemnt car entrainé sur sousData
            # soit une seule paire
            # TODO Q2B
            # Utilisez la grille que vous avez créée plus haut
            # pour afficher les régions de décision, de même
            # que les points colorés selon leur vraie classe

            # PATCH POUR LES LEGENDES...... Je crée un point en bas a gauche de
            # chaque graphe afin de pouvoir afficher la bonne légende de la
            # bonne couleur car je n'ai qu'un seul scatter de data.
            # J'aurais aussi pu faire 3 scatter différents mais cette méthode
            # est bien plus rapide.
            scat = []
            for color in colors:
                scat.append(subfig.scatter(data.data[:, f1].min(),
                                           data.data[:, f2].min(), alpha=1,
                                           c=color, s=20))

            # On affiche les prédictions de la grille afin d'avoir les zones
            # de couleurs.
            subfig.scatter(absc, ordon, alpha=0.05, s=20,
                           c=[colors[i] for i in clf.predict(
                               numpy.c_[absc.ravel(), ordon.ravel()])])
            subfig.scatter(data.data[:, f1], data.data[:, f2], alpha=1, s=20,
                           c=[colors[i] for i in clf.predict(sousData)])
            subfig.set_title(clf.__class__.__name__ + " Err = " +
                             str("%.2f" % round(err, 2)), fontsize=20)
            fig.legend(tuple(scat), tuple(data.target_names),
                       loc="lower center", ncol=5, fontsize=20)

            if err > ERRMAX_Q2B:
                print("[ATTENTION] Votre code pour la question 2B ne produit \
                pas les performances attendues! " +
                      "Le taux d'erreur maximal attendu est de {0:.3f}, mais \
                      l'erreur rapportée dans votre code est de\
                       {1:.3f}!".format(ERRMAX_Q2B, err))

        _times.append(time.time())
        checkTime(TMAX_Q2B, "2B")
        fig.suptitle(str(data.feature_names[f1])[0:-5]
                     + " & " +
                     str(data.feature_names[f2])[0:-5], fontsize=20)

        # On affiche les graphiques
        pyplot.show()

    # ########################################################################
    # ############################  QUESTION 2 C #############################
    # ########################################################################
    # Question 2C
    # Note : Q2C (i) peut être répondue en utilisant le code précédent
    ClassifierQc = QuadraticDiscriminantAnalysis()

    avgError = 0.0
    for i in range(10):
        X, y = data.data, data.target
        ClassifierQc.fit(X, y)
        avgError += 1 - ClassifierQc.score(X, y)
    avgError /= 10
    print("Erreur : ", avgError)

    _times.append(time.time())
    # TODO Q2Cii
    # écrivez ici le code permettant de partitionner les données en jeux
    # d'entraînement / de validation et de tester la performance du classifieur
    # mentionné dans l'énoncé
    # Vous devez répéter cette mesure 10 fois avec des partitions différentes
    # Stockez l'erreur moyenne sur ces 10 itérations dans une variable nommée
    # avgError

    avgError = 0.0
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target,
            test_size=0.5, train_size=0.5)
        ClassifierQc.fit(X_train, y_train)
        avgError += 1 - ClassifierQc.score(X_test, y_test)
    avgError /= 10
    print("Erreur 50-50 : ", avgError)
    _times.append(time.time())
    checkTime(TMAX_Q2Cii, "2Cii")
    if avgError > ERRMAX_Q2Cii:
        print("[ATTENTION] Votre code pour la question 2C ii) ne produit pas\
         les performances attendues! " + "Le taux d'erreur maximal attendu est \
         de {0:.3f}, mais l'erreur rapportée dans votre code est de {1:.3f}!"
              .format(ERRMAX_Q2Cii, avgError))

    _times.append(time.time())
    # TODO Q2Ciii
    # écrivez ici le code permettant de déterminer la performance du
    # classifieur avec un K-fold avec k=3.
    # Vous devez répéter le K-folding 10 fois
    # Stockez l'erreur moyenne sur ces 10 itérations dans une variable nommée
    # avgError
    avgError = 0.0
    for i in range(10):
        rkf = RepeatedKFold(n_splits=3, n_repeats=1)
        for train_index, test_index in rkf.split(data.data):
            X_train, X_test = data.data[train_index], data.data[test_index]
            y_train, y_test = data.target[train_index], data.target[test_index]
            ClassifierQc.fit(X_train, y_train)
            avgError += 1 - ClassifierQc.score(X_test, y_test)
    avgError /= 30
    print("Erreur K-Fold : ", avgError)
    _times.append(time.time())
    checkTime(TMAX_Q2Ciii, "2Ciii")

    if avgError > ERRMAX_Q2Ciii:
        print("[ATTENTION] Votre code pour la question 2C iii) ne produit pas\
         les performances attendues! " + "Le taux d'erreur maximal attendu est \
         de {0:.3f}, mais l'erreur rapportée dans votre code est de {1:.3f}!"
              .format(ERRMAX_Q2Ciii, avgError))

    # ########################################################################
    # ############################  QUESTION 2 D #############################
    # ########################################################################
    # Question 2D

    # TODO Q2D
    # Initialisez ici les différents classifieurs, dans
    # une liste nommée "classifieurs"

    classifieurs = [QuadraticDiscriminantAnalysis(),
                    LinearDiscriminantAnalysis(),
                    GaussianNB(),
                    NearestCentroid()
                    ]

    # Création du jeu de données
    X, y = make_circles(factor=0.3)

    # TODO Q2D
    # Créez ici une grille permettant d'afficher les régions de
    # décision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous être utile ici
    # N'utilisez pas un pas trop petit!
    pas = 100
    a = numpy.linspace(X[:, 0].min(), X[:, 0].max(), pas)
    b = numpy.linspace(X[:, 1].min(), X[:, 1].max(), pas)
    absc, ordon = numpy.meshgrid(a, b)
    # On crée une figure à  plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    _times.append(time.time())
    for clf, subfig in zip(classifieurs, subfigs.reshape(-1)):
        # TODO Q2D
        # Divisez le jeu de données de manière déterministe,
        # puis entraînez le classifieur
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            train_size=0.5,
                                                            random_state=666)
        clf.fit(X_train, y_train)
        # TODO Q2D
        # Obtenez et affichez son erreur (1 - accuracy)
        # Stockez la valeur de cette erreur dans la variable err
        err = 1 - clf.score(X_test, y_test)
        # TODO Q2D
        # Utilisez la grille que vous avez créée plus haut
        # pour afficher les régions de décision, de même
        # que les points colorés selon leur vraie classe
        subfig.set_title(clf.__class__.__name__)
        scat = []
        nb_class = len(set(y))
        colors = "bgrcmykw"[:nb_class]
        for color in colors:
            scat.append(subfig.scatter(X[:, 0].min(), X[:, 1].min(), alpha=1,
                                       c=color, s=20))

        subfig.scatter(absc, ordon, alpha=0.05, s=20,
                       c=[colors[i] for i in clf.predict(
                           numpy.c_[absc.ravel(), ordon.ravel()])])
        subfig.scatter(X[:, 0], X[:, 1], alpha=1, s=20,
                       c=[colors[i] for i in clf.predict(X)])
        # Identification des axes et des méthodes
        subfig.set_xlabel("x")
        subfig.set_ylabel("y")
        subfig.set_title(clf.__class__.__name__ +
                         " Err = " + str("%.2f" % round(err, 2)), fontsize=20)
        fig.legend(tuple(scat), tuple(set(y)),
                   loc="lower center", ncol=5)
    _times.append(time.time())
    checkTime(TMAX_Q2D, "2D")

    pyplot.show()


# N'écrivez pas de code à  partir de cet endroit
