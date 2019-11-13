# -*- coding: utf-8 -*-

#####
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###

import itertools as itt
import numpy as np
import matplotlib.pyplot as plt

class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None



    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        N = len(x_train)
        self.x_train = x_train

        #RBF kernel
        if self.noyau == 'rbf':
            # (x-y).T*(x-y) = x.T*x + y.T*y - 2*x.T*y
            sq_norm = (x_train ** 2).sum(axis=1)# x.T*x
            k    = np.dot(x_train, x_train.T)# x.T*y
            k   *= -2                        # -2*x.T*y
            k   += sq_norm.reshape(-1, 1)    # y.T*y - 2*x.T*y
            k   += sq_norm                   # x.T*x +y.T*y - 2*x.T*y
            k   *= (-1/(2*self.sigma_square))# -||x_i - x_j||^2 / 2*sigma^2
            np.exp(k, k)                     # exp(-||x_i - x_j||^2 / 2*sigma^2)

        #Linear kernel
        if self.noyau == 'lineaire':
            k = np.dot(x_train, x_train.T)

        #Polynomial kernel
        if self.noyau == 'polynomial':
            k  = np.dot(x_train, x_train.T)
            k += self.c
            k  = np.power(k, self.M)

        #Sigmoidal kernel
        if self.noyau == 'sigmoidal':
            k  = np.dot(x_train, x_train.T)
            k *= self.b
            k += self.d
            np.tanh(k, k)

        self.a = np.dot(np.linalg.inv((k + self.lamb*np.identity(N))), t_train)

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        TE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        # RBF kernel
        if self.noyau == 'rbf':
            sq_norm = (x ** 2).sum(axis=0)
            k_x     = np.dot(self.x_train, x.T)
            k_x    *= -2
            k_x    += np.dot(self.x_train, x)
            k_x    += sq_norm
            k_x    *= (-1/(2*self.sigma_square))
            np.exp(k_x, k_x)

        #Linear kernel
        if self.noyau == 'lineaire':
            k_x  = np.dot(self.x_train, x.T)

        #Polynomial kernel
        if self.noyau == 'polynomial':
            k_x  = np.dot(self.x_train, x.T)
            k_x += self.c
            k_x  = np.power(k_x, self.M)

        #Sigmoidal kernel
        if self.noyau == 'sigmoidal':
            k_x  = np.dot(self.x_train, x.T)
            k_x *= self.b
            k_x += self.d
            np.tanh(k_x, k_x)

        y = np.dot(k_x.T, self.a)
        if y>0.5:
            return 1
        return 0


    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (prediction - t)**2

    def validation_croisee(self, x_tab, t_tab, debug=False):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        model_parameters = {
                'lineaire'    : ['lamb', ],
                'rbf'       : ['lamb', 'sigma_sq'],
                'polynomial': ['lamb', 'c', 'M'],
                'sigmoidal' : ['lamb', 'b', 'd']
                }
        def cross_val(kwargs):
            '''
            Implementation of k-fold cross-validation algorithm.
            D_train : shuffled training dataset
            par_values: parameter dictionary
            k: cross validation parameter (number of equaly sized bins)
            '''
            D_train = list(zip(x_tab, t_tab))
            errs = []
            for i in range(0, len(D_train)):
                x_val, t_val = D_train[i]
                D_cv   = D_train[:i] + D_train[i+1:]
                x_cv,   t_cv   = [np.array(x) for x in zip(*D_cv)]
                for p in model_parameters[self.noyau]:
                    setattr(self, p, kwargs[p])
                self.entrainement(x_cv, t_cv)
                t_pred = self.prediction(x_val)
                errs.append(self.erreur(t_val, t_pred))
            return np.mean(errs)

        # grid-search hyperparameters
        grid_size = 20
        par_search_space = {
                'lamb'     : np.logspace(np.log10(1e-9), np.log10(2), grid_size),
                'sigma_sq' : np.logspace(np.log10(1e-9), np.log10(2), grid_size),
                'c'        : np.arange(0, 6),
                'b'        : np.logspace(np.log10(1e-5), np.log10(0.01), grid_size),
                'd'        : np.logspace(np.log10(1e-5), np.log10(0.01), grid_size),
                'M'        : np.arange(2, 7)
                }

        pars = model_parameters[self.noyau]
        args_ls = [dict(zip(pars, x)) for x in itt.product(*[par_search_space[p] for p in pars])]
        meanerr_hyperpars = dict() # mean error as keys and hyperpars as values
        for args in args_ls:
            meanerr_hyperpars[cross_val(args)] = args
        best_hyperpars = meanerr_hyperpars[min(meanerr_hyperpars.keys())]
        if debug:
            print(len(args_ls), best_hyperpars)
        for hyperpar, hyperpar_value in best_hyperpars.items():
            setattr(self, hyperpar, hyperpar_value)
        self.entrainement(x_tab, t_tab)



    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')

        plt.title(self.noyau)

        plt.show()
