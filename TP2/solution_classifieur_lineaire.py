# -*- coding: utf-8 -*-

#####
# Vos Noms (VosMatricules) .~= À MODIFIER =~.
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # AJOUTER CODE ICI
            n   = len(t_train)
            n_2 = sum(t_train) # t=1
            n_1 = n-n_2  # t=0

            p1        = n_1 / (n_1 + n_2)
            p2        = n_2 / (n_1 + n_2)
            mu_1     = np.matrix(sum((x_train.transpose()*(1-t_train)).transpose()) / n_1)
            mu_2     = np.matrix(sum((x_train.transpose()*t_train).transpose()) / n_2)
            s_1      = sum((x-mu_1).transpose()*(x-mu_1) for x,t in zip(x_train, t_train) if t == 0.)/n_1
            s_2      = sum((x-mu_2).transpose()*(x-mu_2) for x,t in zip(x_train, t_train) if t == 1.)/n_2
            sigma    =  (n_1/n)*s_1 + (n_2/n)*s_2 + self.lamb*np.identity(x_train.shape[1])
            sigma_inv = np.linalg.inv(sigma)
            self.w   = np.array(sigma_inv*(mu_1-mu_2).transpose())
            self.w_0 = float((-1/2)*mu_1*sigma_inv*mu_1.transpose() + (-1/2)*mu_2*sigma_inv*mu_2.transpose() + np.log((p1/p2)))


        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            # AJOUTER CODE ICI
            t_train = np.array([1. if t == 1. else -1. for t in t_train])
            x_train = np.array([[1., *x] for x in x_train])
            eta     = 0.001
            iter_max = 1000
            w = np.array([self.w_0, *self.w.transpose()]) #np.random.randn(3)
            k = 0
            misclassified = True
            while misclassified and k <= iter_max:
                k += 1
                misclassified = False
                for n in range(x_train.shape[0]):
                    y = np.matmul(w.transpose(), x_train[n])

                    if y*t_train[n] < 0: # donnee mal classee
                        misclassified = True
                        w = w + eta*t_train[n]*x_train[n]
            self.w_0 = w[0]
            self.w   = w[1:]


        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            # AJOUTER CODE ICI
            clf = Perceptron(eta0=0.001, penalty='l2')
            clf.fit(x_train, t_train)
            self.w_0 = clf.intercept_
            self.w   = clf.coef_.transpose()


        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        # AJOUTER CODE ICI
        y = self.w_0 + np.matmul(self.w.transpose(), x)
        if y > 0:
            return 1
        return 0

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # AJOUTER CODE ICI
        if t != prediction:
            return 1
        return 0

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')
        plt.savefig('figure.png') #remove before submitting

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
