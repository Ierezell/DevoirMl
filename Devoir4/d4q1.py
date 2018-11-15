###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 4, Question 1
#
# #############################################################################
# ############################# INSTRUCTIONS ##################################
# #############################################################################
#
# - Repérez les commentaires commenà§ant par TODO : ils indiquent une tâche
#       que vous devez effectuer.
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision
from d4utils import CODES_DE_SECTION
from d4utils import VolcanoesConv
from d4utils import VolcanoesDataset, VolcanoesLoader
from d4utils import compute_accuracy, compute_confusion_matrix


# TODO Logistique
# Mettre 'BACC' ou 'GRAD'
SECTION = 'GRAD'

# TODO Logistique
# Mettre son numéro d'équipe ici
NUMERO_EQUIPE = 15

# Crée la random seed
RANDOM_SEED = CODES_DE_SECTION[SECTION] + NUMERO_EQUIPE


class VolcanoesNet(nn.Module):
    """
    Cette classe définit un réseau
    plainement convolutionnel simple
    permettant de classifier des images
    satellite de Venus.
    """

    def __init__(self):
        super().__init__()

        # TODO Q1A
        # Définir ici les couches de convolution
        # comme il est décrit dans l'énoncé du
        # devoir
        self.C1 = VolcanoesConv(1, 32, kernel=5)
        self.C2 = VolcanoesConv(32, 64, kernel=3)
        self.C3 = VolcanoesConv(64, 64, kernel=3)
        self.C4 = VolcanoesConv(64, 64, kernel=3)
        self.C5 = VolcanoesConv(64, 64, kernel=3)

        # TODO Q1A
        # Définir les couches de normalisation
        # permettant de maintenir les valeurs
        # du réseau à des valeurs raisonnables
        self.B1 = nn.BatchNorm2d(32)
        self.B2 = nn.BatchNorm2d(64)
        self.B3 = nn.BatchNorm2d(64)
        self.B4 = nn.BatchNorm2d(64)
        self.B5 = nn.BatchNorm2d(64)

        # TODO Q1A
        # Déninir la couche linéaire de sortie
        self.lin1 = nn.Linear(64, 1)

    def conv_forward(self, x):
        # TODO Q1B
        # Écrire les lignes de code pour l'inférence
        # des couches de convolution, avant l'average
        # pooling
        conv_batch_relu1 = F.relu(self.B1(self.C1(x)))
        conv_batch_relu2 = F.relu(self.B2(self.C2(conv_batch_relu1)))
        conv_batch_relu3 = F.relu(self.B3(self.C3(conv_batch_relu2)))
        conv_batch_relu4 = F.relu(self.B4(self.C4(conv_batch_relu3)))
        y = F.relu(self.B5(self.C5(conv_batch_relu4)))
        return y

    def forward(self, x):
        # Sélectionne la taille de l'entrée
        batch_size = x.size()[0]

        # Exécute la partie convolution
        y = self.conv_forward(x)

        # Fait un average pooling
        y = y.view(batch_size, 64, -1).mean(dim=2)

        return torch.sigmoid(self.lin1(y))


if __name__ == '__main__':
    # Définit la seed
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Des effets stochastiques peuvent survenir
    # avec cudnn, même si la seed est activée
    # voir le thread: https://bit.ly/2QDNxRE
    torch.backends.cudnn.deterministic = True

    # Définit si cuda est utilisé ou non
    # mettre cuda pour utiliser un GPU
    device = 'cpu'

    # Définit les paramètres d'entraînement
    # Nous vous conseillons ces paramètres.
    # Cependant, vous pouvez les changer
    nb_epoch = 10
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32

    # Charge les données d'entraînement et de test
    train_set = VolcanoesDataset('data/Volcanoes_train.pt.gz')
    test_set = VolcanoesDataset('data/Volcanoes_test.pt.gz')

    # Crée le dataloader d'entraînement
    train_loader = VolcanoesLoader(train_set, batch_size=batch_size,
                                   balanced=True, random_seed=RANDOM_SEED)
    test_loader = VolcanoesLoader(test_set, batch_size=batch_size,
                                  balanced=True, random_seed=RANDOM_SEED)

    # TODO Q1C
    # Instancier un réseau VolcanoesNet
    # dans une variable nommée "model"
    model = VolcanoesNet()
    # Tranfert le réseau au bon endroit
    model.to(device)

    # TODO Q1C
    # Instancier une fonction d'erreur BinaryCrossEntropy
    # et la mettre dans une variable nommée criterion
    criterion = torch.nn.BCELoss()
    # TODO Q1C
    # Instancier l'algorithme d'optimisation SGD
    # Ne pas oublier de lui donner les hyperparamètres
    # d'entraînement : learning rate et momentum!
    optimizer = SGD(model.parameters(), lr=learning_rate,
                    momentum=momentum)
    # TODO Q1C
    # Mettre le réseau en mode entraînement
    model.train()
    # TODO Q1C
    # Remplir les TODO dans la boucle d'entraînement
    for i_epoch in range(nb_epoch):
        start_time, train_losses = time.time(), []
        for i_batch, batch in enumerate(train_loader):
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)

            # TODO Q1C
            # Mettre les gradients à  zéro
            optimizer.zero_grad()
            # TODO Q1C
            # Calculer:
            # 1. l'inférence dans une variable "predictions"
            # 2. l'erreur dans une variable "loss"
            predictions = model(images)
            loss = criterion(predictions, targets)
            # TODO Q1C
            # Rétropropager l'erreur et effectuer
            # une étape d'optimisation
            loss.backward()
            optimizer.step()

            # Ajoute le loss de la batch
            train_losses.append(loss.item())

        print(' [-] epoch {:4}/{:}, train loss {:.6f} in {:.2f}s'.format(
            i_epoch+1, nb_epoch, np.mean(train_losses),
            time.time()-start_time))

        # sauvegarde le réseau
        torch.save(model.state_dict(), 'volcanoes_model.pt')

    # affiche le score à  l'écran
    test_acc = compute_accuracy(model, test_loader, device)
    print(' [-] test acc. {:.6f}%'.format(test_acc * 100))

    # affiche la matrice de confusion à  l'écran
    matrix = compute_confusion_matrix(model, test_loader, device)
    print(' [-] conf. mtx. : \n{}'.format(matrix))
