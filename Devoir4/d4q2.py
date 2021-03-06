###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 4, Question 2
#
# #############################################################################
# ############################# INSTRUCTIONS ##################################
# #############################################################################
#
# - Repérez les commentaires commenÃ§ant par TODO : ils indiquent une tÃ¢che
#       que vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, Ã  la fonction
#       checkTime, ni aux conditions vérifiant le bon fonctionnement de votre
#       code. Ces structures vous permettent de savoir rapidement si vous ne
#       respectez pas les requis minimum pour une question en particulier.
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer
#       la note de zéro (0) pour la partie implémentation!
#
###############################################################################

import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torchvision.transforms as T

from d4utils import CODES_DE_SECTION
from d4utils import compute_accuracy


# TODO Logistique
# Mettre 'BACC' ou 'GRAD'
SECTION = 'GRAD'

# TODO Logistique
# Mettre son numéro d'équipe ici
NUMERO_EQUIPE = 15

# Crée la random seed
RANDOM_SEED = CODES_DE_SECTION[SECTION] + NUMERO_EQUIPE


class LegoNet(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()

        # Crée le réseau de neurone pré-entraÃ®né
        self.model = resnet18(pretrained=pretrained)

        # RécupÃ¨re le nombre de neurones avant
        # la couche de classification
        dim_before_fc = self.model.fc.in_features

        # TODO Q2A
        # Changer la derniÃ¨re fully-connected layer
        # pour avoir le bon nombre de neurones de
        # sortie

        if pretrained:
            # TODO Q2A
            # Geler les paramÃ¨tres qui ne font pas partie
            # de la derniÃ¨re couche fc
            # Conseil: utiliser l'itérateur named_parameters()
            # et la variable requires_grad

    def forward(self, x):
        # TODO Q2B
        # Appeler la fonction forward du réseau
        # pré-entraÃ®né (resnet18) de LegoNet

        return y


if __name__ == '__main__':
    # Définit la seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Des effets stochastiques peuvent survenir
    # avec cudnn, mÃªme si la seed est activée
    # voir le thread: https://bit.ly/2QDNxRE
    torch.backends.cudnn.deterministic = True

    # TODO Q2D
    # Faire rouler le code une fois sans pré-entrainement
    # et l'autre fois avec pré-entraÃ®nement
    pretrained = False

    # Définit si cuda est utilisé ou non
    # mettre cuda pour utiliser un GPU
    device = 'cpu'

    # Définit les paramÃ¨tres d'entraÃ®nement
    # Nous vous conseillons ces paramÃ¨tres. Or, vous pouvez
    # les changer
    nb_epoch = 1
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 64

    # Définit les transformations nécessaires pour
    # le chargement du dataset
    totensor = T.ToTensor()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    composition = T.Compose([totensor, normalize])

    # Charge le dataset d'entraÃ®nement
    train_set = ImageFolder('data/lego_data/train', transform=composition)

    # Crée un générateur aléatoire avec la seed
    context_random = random.Random(RANDOM_SEED)

    # Selectionne 10% du jeu de test aléatoirement pour alléger le calcul
    test_set = ImageFolder('data/lego_data/test', transform=composition)
    idx = context_random.sample(
        range(len(test_set)), k=int(0.1 * len(test_set)))
    test_set.samples = [test_set.samples[i] for i in idx]

    # TODO Q2C
    # Créer les dataloader pytorch avec la
    # classe DataLoader de pytorch

    # TODO Q2C
    # Instancier un réseau LegoNet
    # dans une variable nommée "model"

    # Tranfert le réseau au bon endroit
    model.to(device)

    # TODO Q2C
    # Instancier une fonction d'erreur CrossEntropyLoss
    # et la mettre dans une variable nommée criterion

    # TODO Q2C
    # Instancier l'algorithme d'optimisation SGD
    # Conseil: Filtrer les paramÃ¨tres non-gelés!
    # Ne pas oublier de lui donner les hyperparamÃ¨tres
    # d'entraÃ®nement : learning rate et momentum!

    # TODO Q2C
    # Mettre le réseau en mode entraÃ®nement

    # RécupÃ¨re le nombre total de batch pour une epoch
    total_batch = len(train_loader)

    # TODO Q2C
    # Remplir les TODO dans la boucle d'entraÃ®nement
    for i_epoch in range(nb_epoch):

        start_time, train_losses = time.time(), []
        for i_batch, batch in enumerate(train_loader):
            images, targets = batch

            images = images.to(device)
            targets = targets.to(device)

            # TODO Q2C
            # Mettre les gradients Ã  zéro

            # TODO Q2C
            # Calculer:
            # 1. l'inférence dans une variable "predictions"
            # 2. l'erreur dans une variable "loss"

            # TODO Q2C
            # Rétropropager l'erreur et effectuer
            # une étape d'optimisation

            # Ajoute le loss de la batch
            train_losses.append(loss.item())

            print(' [-] batch {:4}/{:4} since {:.2f}s'.format(
                i_batch+1, total_batch, time.time()-start_time))

        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch+1, nb_epoch, np.mean(train_losses), time.time()-start_time))

        # sauvegarde le réseau
        if pretrained:
            torch.save(model.state_dict(), 'lego_model_pretrained.pt')
        else:
            torch.save(model.state_dict(), 'lego_model_not_pretrained.pt')

    # affiche le score Ã  l'écran
    test_acc = compute_accuracy(model, test_loader, device)
    if pretrained:
        print(' [-] pretrained test acc. {:.6f}%'.format(test_acc * 100))
    else:
        print(' [-] not pretrained test acc. {:.6f}%'.format(test_acc * 100))
