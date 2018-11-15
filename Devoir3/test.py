# -*- coding: utf-8 -*-
# ###########################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 3
#
# ###########################################################################
# ############################# INSTRUCTIONS ################################
# ###########################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tà¢che
# que vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, à  la fonction
#       checkTime, ni aux conditions vérifiant le bon fonctionnement de votre
#       code. Ces structures vous permettent de savoir rapidement si vous ne
#       respectez pas les requis minimum pour une question en particulier.
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer
#       la note de zéro (0) pour la partie implémentation!
#
#############################################################################

import time
import numpy

from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from matplotlib import pyplot


# Fonctions utilitaires liées à  l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("""[ATTENTION] Votre code pour la question {0} met trop de temps
            à s'exécuter! """.format(question) + """Le temps maximum permis
            est de {0: .4f} secondes, mais votre code a requis {1: .4f}
            secondes!""".format(maxduration, duration) + """Assurez-vous que
            vous ne faites pas d'appels bloquants (par exemple à  show())
            dans cette boucle!""")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_FIT = 2.0
TMAX_EVAL = 3.0


# Ne modifiez rien avant cette ligne!


# Question 3B
# Implémentation du discriminant à  noyau
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma):
        # Cette fonction est déjà  codée pour vous, vous n'avez qu'à  utiliser
        # les variables membres qu'elle définit dans les autres fonctions de
        # cette classe.
        # Lambda et sigma sont définis dans l'énoncé.
        self.lambda_ = lambda_
        self.sigma = sigma

    def fit(self, X, y):
        if set(y) == {0, 1}:
            y = (y * 2) - 1
        elif set(y) != {-1, 1}:
            raise Exception("The target must be {0,1} or {-1,1}")
        # Implémentez la fonction d'entraînement du classifieur, selon
        # les équations que vous avez développées dans votre rapport.
        y = numpy.array(y)
        X = numpy.array(X)
        # TODO Q3B
        # Vous devez écrire une fonction nommée evaluateFunc,
        # qui reçoit un seul argument en paramètre, qui correspond aux
        # valeurs des paramètres pour lesquels on souhaite connaître
        # l'erreur et le gradient d'erreur pour chaque paramètre.
        # Cette fonction sera appelée à  répétition par l'optimiseur
        # de scipy, qui l'utilisera pour minimiser l'erreur et obtenir
        # un jeu de paramètres optimal.

        Kgauss = numpy.exp(-(cdist(X, X, 'minkowski', p=2.)) / (self.sigma**2))

        def evaluateFunc(hypers):
            w0, *alphas = hypers

            def h(T): return [numpy.sum(
                alphas*y*Kgauss[t], axis=0)+w0 for t in T]

            # def h(T): return [numpy.dot(alphas*y,Kgauss[T])+w0]

            ind_mal_classe = numpy.where(y*h(list(range(len(X)))) < 1)[0]
            err = numpy.sum(
                1 - y[ind_mal_classe]*h(ind_mal_classe)) +\
                self.lambda_*numpy.sum(alphas)

            gradalpha = -(y*numpy.dot(y[ind_mal_classe],
                                      Kgauss[ind_mal_classe])) +\
                self.lambda_

            gradW0 = [numpy.sum(y[ind_mal_classe])]
            grad = numpy.concatenate((gradW0, gradalpha))
            return err, grad

        # TODO Q3B
        # Initialisez aléatoirement les paramètres alpha et omega0
        # (l'optimiseur requiert un "initial guess", et nous ne pouvons pas
        # simplement n'utiliser que des zéros pour différentes raisons).
        alpha = numpy.random.rand(X.shape[0])
        # alpha = numpy.zeros(X.shape[0])
        w0 = numpy.random.rand(1)
        # Stochez ces valeurs initiales aléatoires dans un array numpy nommé
        # "params"
        params = numpy.concatenate((w0, alpha), axis=0)
        # Déterminez également les bornes à  utiliser sur ces paramètres
        # et stockez les dans une variable nommée "bounds".
        # Indice : les paramètres peuvent-ils avoir une valeur maximale (au-
        # dessus de laquelle ils ne veulent plus rien dire)? Une valeur
        # minimale? Référez-vous à  la documentation de fmin_l_bfgs_b
        # pour savoir comment indiquer l'absence de bornes.
        bounds = [(None, None)]+[(0, None) for i in range(len(alpha))]
        # à€ ce stade, trois choses devraient être définies :
        # - Une fonction d'évaluation nommée evaluateFunc, capable de retourner
        #   l'erreur et le gradient d'erreur pour chaque paramètre pour une
        #   configuration de paramètres alpha et omega_0 donnée.
        # - Un tableau numpy nommé params de même taille que le nombre de
        #   paramètres à  entraîner.
        # - Une liste nommée bounds contenant les bornes que l'optimiseur doit
        #   respecter pour chaque paramètre
        # On appelle maintenant l'optimiseur avec ces informations et on stocke
        # les valeurs optimisées dans params
        _times.append(time.time())
        params, minval, infos = fmin_l_bfgs_b(
            evaluateFunc, params, bounds=bounds, factr=1e12, epsilon=2)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        # On affiche quelques statistiques
        print("Entraînement terminé après {it} itérations et "
              "{calls} appels à  evaluateFunc".format(it=infos['nit'],
                                                      calls=infos['funcalls']))
        print("\tErreur minimale : {:.5f}".format(minval))
        print("\tL'algorithme a convergé" if infos['warnflag'] == 0 else "\t \
                 L'algorithme n'a PAS convergé")
        print("\tGradients des paramètres à  la convergence (ou à l'épuisement\
                des ressources) :")
        # print(infos['grad'])

        # TODO Q3B
        # Stockez les paramètres optimisés de la façon suivante
        # - Le vecteur alpha dans self.alphas
        # - Le biais omega0 dans self.w0
        self.alphas = params[1:]
        self.w0 = params[0]
        # On retient également le jeu d'entraînement, qui pourra
        # vous être utile pour les autres fonctions à  implémenter
        self.X, self.y = X, y

    def predict(self, X):
        # TODO Q3B
        # Implémentez la fonction de prédiction
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # et que les variables membres alphas, w0, X et y existent.
        # N'oubliez pas que ce classifieur doit retourner -1 ou 1
        Kgauss = numpy.exp(-(cdist(self.X, X, 'minkowski',
                                   p=2.)) / (self.sigma**2))
        predictions = []
        for t in range(len(X)):
            ht = numpy.sum([self.y[s] * self.alphas[s] * Kgauss[s, t]
                            for s in range(len(self.X))]) + self.w0
            predictions.append(int(numpy.sign(ht)))
        return predictions

    def score(self, X, y):
        # TODO Q3B
        # Implémentez la fonction retournant le score (accuracy)
        # du classifieur sur les données reçues en argument.
        # Vous pouvez supposer que fit() a préalablement été exécuté
        # Indice : réutiliser votre implémentation de predict() réduit de
        # beaucoup la taille de cette fonction!
        predictions = self.predict(X)
        return numpy.sum([predictions[i] == y[i]
                          for i in range(len(y))]) / len(y)


if __name__ == "__main__":
    # Question 3B

    # TODO Q3B
    # Créez le jeu de données à  partir de la fonction make_moons, tel que
    # demandé dans l'énoncé
    # N'oubliez pas de vous assurer que les valeurs possibles de y sont
    # bel et bien -1 et 1, et non 0 et 1!
    X, y = make_moons(n_samples=1000, shuffle=True,
                      noise=0.3, random_state=None)
    y = (y * 2) - 1
    # TODO Q3B
    # Séparez le jeu de données en deux parts égales, l'une pour l'entraînement
    # et l'autre pour le test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=666, shuffle=True)

    listSigma = [0.1*x for x in range(1, 10)]
    listLambda = [0.01*x for x in range(1, 10)] +\
        [0.1*x for x in range(1, 10)] + [1, 5, 10]
    listDaN = []
    ErrorDaN = {(l, s): 0 for l in listLambda for s in listSigma}
    for l in listLambda:
        for s in listSigma:
            listDaN.append(DiscriminantANoyau(lambda_=l, sigma=s))

    print("Nombre de classifieurs : ", len(listDaN))
    for i in range(len(listDaN)):
        print("\n\nPARAM L & S", listDaN[i].lambda_, listDaN[i].sigma)
        print("Classificateur : ", i, "/", len(listDaN))
        listDaN[i].fit(X_train, y_train)
        ErrorDaN[listDaN[i].lambda_, listDaN[i].sigma] = 1 -\
            listDaN[i].score(X_train, y_train)
        print("Error", ErrorDaN[listDaN[i].lambda_, listDaN[i].sigma])
    minErrorDaN = min([ErrorDaN[i] for i in ErrorDaN.keys()])
    argMinDaN = list(ErrorDaN.keys())[list(
        ErrorDaN.values()).index(minErrorDaN)]

    for k, v in ErrorDaN.items():
        print(k, v)
    print("argmin Lambda, Sigma = ", argMinDaN)
    print("minErrorDaN", minErrorDaN)
    print("Good parameters", [k for k, v in ErrorDaN.items() if v < 0.10])

    fig, sfig = pyplot.subplots(1, 1, sharex=True, sharey=True)
    err = sfig.scatter([list(ErrorDaN.keys())[i][0]
                        for i in range(len(list(ErrorDaN.keys())))],
                       [list(ErrorDaN.keys())[i][1]
                        for i in range(len(list(ErrorDaN.keys())))],
                       s=80, c=list(ErrorDaN.values()))
    sfig.set_xlabel("lambda")
    sfig.set_ylabel("Sigma")
    pyplot.colorbar(err, format='Err %.2f')
    pyplot.show()

    _times.append(time.time())
    # TODO Q3B
    # Une fois les paramètres lambda et sigma de votre classifieur optimisés,
    # créez une instance de ce classifieur en utilisant ces paramètres
    # optimaux, et calculez sa performance sur le jeu de test.
    lambda_opti, sigma_opti = argMinDaN
    clfopti = DiscriminantANoyau(lambda_opti, sigma_opti)
    clfopti.fit(X_test, y_test)
    ScoreDiscNoy = clfopti.score(X_test, y_test)
    print("Score DiscLinGauss: ", ScoreDiscNoy)
    # TODO Q3B
    # Créez ici une grille permettant d'afficher les régions de
    # décision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous être utile ici
    # Par la suite, affichez les régions de décision dans la même figure
    # que les données de test.
    # Note : utilisez un pas de 0.02 pour le meshgrid
    pas = 0.2
    x1 = numpy.arange(X[:, 0].min()-0.5, X[:, 0].max()+0.5, pas)
    x2 = numpy.arange(X[:, 1].min()-0.5, X[:, 1].max()+0.5, pas)
    absc, ordon = numpy.meshgrid(x1, x2)

    fig, sfig = pyplot.subplots(1, 1, sharex=True, sharey=True)
    prediction = clfopti.predict(numpy.c_[absc.ravel(), ordon.ravel()])
    colors = ["bgr"[i] for i in prediction]
    Z = numpy.array(prediction)
    Z = Z.reshape(absc.shape)
    sfig.contourf(absc, ordon, Z, alpha=0.8)
    # sfig.scatter(absc, ordon, alpha=0.5, s=20, c=colors)
    # sfig.scatter(X_train[:, 0], X_train[:, 1], marker='*',
    #              c=["bgr"[i] for i in y_train],alpha=0.2)
    sfig.scatter(X_test[:, 0], X_test[:, 1], marker='s',
                 color=["bgr"[i] for i in y_test], alpha=0.5)

    sfig.set_title("Erreur : " + str("%.2f" % round(1 - ScoreDiscNoy,
                                                    2)), fontsize=20)
    # On affiche la figure
    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation")
    pyplot.show()
# N'écrivez pas de code à  partir de cet endroit
