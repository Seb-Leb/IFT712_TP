# -*- coding: utf-8 -*-

#####
# Jeremie Beliveau-Lefebvre (04494470)
# Sebastien Leblanc         (18206273)
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # AJOUTER CODE ICI
        phi_x = x
        if self.M > 0:
            phi_x = [0,]+[x**i for i in range(0,self.M+1)]

        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI
        # shuffle training set
        D_train = list(zip(X,t))
        random.shuffle(D_train)
        # divide dataset into k equal portions
        def k_fold_cross_val(D_train, m, k=10):
            '''
            Implementation of k-fold cross-validation algorythm.
            D_train : shuffled training dataset
            m: parameter for regression
            k: cross validation parameter (number of equaly sized bins)
            '''
            len_D_train = len(D_train)
            bin_size    = int(len_D_train / k)
            for i in range(0, len_D_train, bin_size):
                X_val, t_val = list(zip(*D_train[i:i+bin_size]))
                D_cv   = D_train[:i]
                MSEs   = []
                if i+bin_size < len_D_train:
                    D_cv += D_train[i+bin_size:]
                X_cv,   t_cv   = list(zip(*D_cv))
                self.M = m
                self.entrainement(X_cv, t_cv)
                t_pred = np.array([self.prediction(x) for x in X_val])
                MSE = np.mean([self.erreur(t_v, t_p) for t_v, t_p in zip(t_val, t_pred)])
                MSEs.append(MSE)
            return np.mean(MSEs)

        cv_scores = {}
        for m in range(1,11):
            score = k_fold_cross_val(D_train, m)
            cv_scores[score] = m

        self.M = cv_scores[min(cv_scores.keys())]

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI

        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = np.array([self.fonction_base_polynomiale(x) for x in X])
        self.w = [0, 1]

        if using_sklearn:
            clf = linear_model.Ridge(alpha=self.lamb)
            clf.fit(phi_x,t)
            self.w = clf.intercept_ + clf.coef_
        else:
            a = (self.lamb*np.identity(phi_x.shape[1]) + np.matmul(phi_x.transpose(),phi_x))
            b = np.matmul(phi_x.transpose(),t)
            self.w = np.linalg.solve(a,b)




    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI
        phi_x = self.fonction_base_polynomiale(x)
        y = np.matmul(self.w.transpose(), phi_x)
        return y

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI
        r2 = (t-prediction)**2
        return r2
