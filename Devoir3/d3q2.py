# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 2
#
# #############################################################################
# ############################# INSTRUCTIONS ##################################
# #############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tâche
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

# import itertools
import time
import numpy
import warnings
from io import BytesIO
from http.client import HTTPConnection

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fonctions utilitaires liées à l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("""[ATTENTION] Votre code pour la question {0} met trop de temps
                à s'exécuter! """.format(question) + """Le temps maximum permis
                est de {0: .4f} secondes, mais votre code a requis {1: .4f}
                secondes!""".format(maxduration, duration) + """Assurez-vous
                que vous ne faites pas d'appels bloquants
                (par exemple à  show()) dans cette boucle!""")


# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_KNN = 40
TMAX_SVM = 200
TMAX_PERCEPTRON = 400
TMAX_EVAL = 80


def fetchPendigits():
    """
    Cette fonction télécharge le jeu de données pendigits et le
    retourne sous forme de deux tableaux numpy. Le premier élément
    retourné par cette fonction est un tableau de 10992x16 qui
    contient les samples; le second élément est un vecteur de 10992
    qui contient les valeurs cible (target).
    """
    host = 'vision.gel.ulaval.ca'
    url = '/~cgagne/enseignement/apprentissage/A2018/travaux/ucipendigits.npy'
    connection = HTTPConnection(host, port=80, timeout=10)
    connection.request('GET', url)

    rep = connection.getresponse()
    if rep.status != 200:
        print("ERREUR : impossible de télécharger le jeu de données UCI \
                Pendigits! Code d'erreur {}".format(rep.status))
        print("Vérifiez que votre ordinateur est bien connecté à  Internet.")
        return
    stream = BytesIO(rep.read())
    dataPendigits = numpy.load(stream)
    return dataPendigits[:, :-1].astype('float32'), dataPendigits[:, -1]

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 2B

    # TODO Q2B
    # Chargez le jeu de données Pendigits. Utilisez pour cela la fonction
    # fetchPendigits fournie. N'oubliez pas de normaliser
    # les données d'entrée entre 0 et 1 pour toutes les dimensions.
    # Notez finalement que fetch_openml retourne les données d'une manière
    # différente des fonctions load_*, assurez-vous que vous utilisez
    # correctement les données et qu'elles sont du bon type.
    X, y = fetchPendigits()
    X = minmax_scale(X)
    # TODO Q2B
    # Séparez le jeu de données Pendigits en deux sous-jeux:
    # entraînement (5000) et test (reste des données). Pour la suite du code,
    # rappelez-vous que vous ne pouvez PAS vous servir du jeu de test pour
    # déterminer la configuration d'hyper-paramètres la plus performante.
    # Ce jeu de test ne doit être utilisé qu'à  la toute fin, pour rapporter
    # les résultats finaux en généralisation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-(5000/len(X)), random_state=666, shuffle=True)
    # TODO Q2B
    # Pour chaque classifieur :
    # - k plus proches voisins,
    # - SVM à  noyau gaussien,
    # - Perceptron multicouche,
    # déterminez les valeurs optimales des hyper-paramètres à  utiliser.
    # Suivez les instructions de l'énoncé quant au nombre d'hyper-paramètres à
    # optimiser et n'oubliez pas d'expliquer vos choix d'hyper-paramètres
    # dans votre rapport.
    # Vous êtes libres d'utiliser la méthodologie que vous souhaitez, en autant
    # que vous ne touchez pas au jeu de test.
    #
    # Note : optimisez les hyper-paramètres des différentes méthodes dans
    # l'ordre dans lequel ils sont énumérés plus haut, en insérant votre code
    # d'optimisation entre les commentaires le spécifiant

    _times.append(time.time())
    # TODO Q2B
    # Optimisez ici la paramétrisation du kPP
    listK = list(range(1, 10))
    listKnn = list(map(lambda k: KNeighborsClassifier(
        n_neighbors=k, weights='distance'), listK))
    ErrorKnn = {listKnn[i].n_neighbors: 0 for i in range(len(listKnn))}
    print("Nombre de classifieurs : ", len(listKnn))

    argMinKnn = 0
    Nb_boucle = 20
    Error = {listKnn[i].n_neighbors: 0 for i in range(len(listKnn))}
    for _ in range(Nb_boucle):
        kf = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            X_train_train = X_train[train_index]
            X_train_test = X_train[test_index]
            y_train_train = y_train[train_index]
            y_train_test = y_train[test_index]
            for i in range(len(listKnn)):
                listKnn[i].fit(X_train_train, y_train_train)
                ErrorKnn[listKnn[i].n_neighbors] += 1 - \
                    listKnn[i].score(X_train_test, y_train_test)
        ErrorKnn = {k: v / kf.get_n_splits(X_train)
                    for k, v in ErrorKnn.items()}
        minErrorKnn = min([ErrorKnn[i] for i in ErrorKnn.keys()])
        argmin = list(ErrorKnn.keys())[
            list(ErrorKnn.values()).index(minErrorKnn)]
        argMinKnn += argmin
    argMinKnn /= Nb_boucle
    print("Error", ErrorKnn)
    print("argminlast K = ", argmin)
    print("argminmoy K = ", argMinKnn)
    argMinKnn = round(argMinKnn)
    print("Param retenu K = ", argMinKnn)
    print("AccKnn", 1-minErrorKnn)
    _times.append(time.time())
    checkTime(TMAX_KNN, "K plus proches voisins")
    # TODO Q2B
    # Optimisez ici la paramétrisation du SVM à  noyau gaussien
    MatrixSigma = numpy.array(pairwise_distances(X_train))
    SigmaMin = (MatrixSigma + numpy.identity(MatrixSigma.shape[0])*1e10).min()
    SigmaParam = numpy.array([SigmaMin*(2**i) for i in range(6)])
    listGamma = 1.0/(2*(SigmaParam**2))
    listC = [10**n for n in range(-5, 6)]
    print(listC)
    print(listGamma)
    listSVC = []
    ErrorSVC = {(c, g): 0 for c in listC for g in listGamma}

    for g in listGamma:
        for c in listC:
            listSVC.append(SVC(kernel='rbf', C=c, gamma=g))

    print("Nombre de classifieurs : ", len(listSVC))

    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X_train):
        X_train_train = X_train[train_index]
        X_train_test = X_train[test_index]
        y_train_train = y_train[train_index]
        y_train_test = y_train[test_index]
        for i in range(len(listSVC)):
            listSVC[i].fit(X_train_train, y_train_train)
            ErrorSVC[listSVC[i].C, listSVC[i].gamma] += 1 - \
                listSVC[i].score(X_train_test, y_train_test)
    ErrorSVC = {k: v / kf.get_n_splits(X_train) for k, v in ErrorSVC.items()}
    minErrorSVC = min([ErrorSVC[i] for i in ErrorSVC.keys()])
    argMinSVC = list(ErrorSVC.keys())[list(
        ErrorSVC.values()).index(minErrorSVC)]
    print("Error", ErrorSVC)
    print("argmin C, g = ", argMinSVC)
    print("AccSvc", 1-minErrorSVC)

    _times.append(time.time())
    checkTime(TMAX_SVM, "SVM")
    # TODO Q2B
    # Optimisez ici la paramétrisation du perceptron multicouche
    # Note : il se peut que vous obteniez ici des "ConvergenceWarning"
    # Ne vous en souciez pas et laissez le paramètre max_iter à  sa
    # valeur suggérée dans l'énoncé (100)

    Nfeat = len(X[0])
    Nsortie = len(set(y_train))
    listHidLay1 = [i*Nfeat for i in range(2, 7)]
    listHidLay2 = [i*Nsortie for i in range(2, 9)]
    listHidLay = [(x, y) for x in listHidLay1 for y in listHidLay2]
    # + [tuple(listHidLay1[i:i+1])
    #               for i in range(len(listHidLay1))]
    print(listHidLay1)
    print(listHidLay2)
    print(listHidLay)
    print("Nombre de classifieurs : ", len(listHidLay))
    listMLP = list(map(lambda HidLay: MLPClassifier(
        hidden_layer_sizes=HidLay), listHidLay))
    ErrorMLP = {
        listMLP[i].hidden_layer_sizes: 0 for i in range(len(listHidLay))}

    kf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in kf.split(X_train):
        X_train_train = X_train[train_index]
        X_train_test = X_train[test_index]
        y_train_train = y_train[train_index]
        y_train_test = y_train[test_index]
        for i in range(len(listMLP)):
            listMLP[i].fit(X_train_train, y_train_train)
            ErrorMLP[listMLP[i].hidden_layer_sizes] += 1 - \
                listMLP[i].score(X_train_test, y_train_test)
    ErrorMLP = {k: v / kf.get_n_splits(X_train) for k, v in ErrorMLP.items()}
    minErrorMLP = min([ErrorMLP[i] for i in ErrorMLP.keys()])
    argMinMLP = list(ErrorMLP.keys())[list(
        ErrorMLP.values()).index(minErrorMLP)]
    print("Error", ErrorMLP)
    print("argmin HidLay1, HidLay 2, HidLay3 = ", argMinMLP)
    print("AccMLP", 1-minErrorMLP)
    _times.append(time.time())
    checkTime(TMAX_PERCEPTRON, "Perceptron")

# TODO Q2B
# Évaluez les performances des meilleures paramétrisations sur le jeu de
# test et rapportez ces performances dans le rapport
    tempsclass = []
    ClassMLP = MLPClassifier(hidden_layer_sizes=argMinMLP)
    ClassSVC = SVC(kernel='rbf', C=argMinSVC[0], gamma=argMinSVC[1])
    ClassKnn = KNeighborsClassifier(n_neighbors=argMinKnn)
    kf = KFold(n_splits=3, shuffle=True)
    ErrorKnn, ErrorSVC, ErrorMLP = 0, 0, 0
    for train_index, test_index in kf.split(X_test):
        X_test_train = X_test[train_index]
        X_test_test = X_test[test_index]
        y_test_train = y_test[train_index]
        y_test_test = y_test[test_index]
        tempsclass.append(time.time())
        ClassMLP.fit(X_test_train, y_test_train)
        tempsclass.append(time.time())
        print("train MLP", tempsclass[-1]-tempsclass[-2])
        ClassSVC.fit(X_test_train, y_test_train)
        tempsclass.append(time.time())
        print("train Svm", tempsclass[-1]-tempsclass[-2])
        ClassKnn.fit(X_test_train, y_test_train)
        tempsclass.append(time.time())
        print("train Knn", tempsclass[-1]-tempsclass[-2])
        ErrorMLP += 1 - ClassMLP.score(X_test_test, y_test_test)
        ErrorSVC += 1 - ClassSVC.score(X_test_test, y_test_test)
        ErrorKnn += 1 - ClassKnn.score(X_test_test, y_test_test)

    print("ErrorMLP : ", ErrorMLP/kf.get_n_splits(X_test))
    print("ErrorSVM : ", ErrorSVC/kf.get_n_splits(X_test))
    print("ErrorKnn : ", ErrorKnn/kf.get_n_splits(X_test))
    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation des modèles")
# N'écrivez pas de code à  partir de cet endroit
