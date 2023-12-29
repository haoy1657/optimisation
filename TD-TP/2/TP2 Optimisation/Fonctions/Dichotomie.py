"""
Created on Sat Oct 17 10:25:15 2020

UE MU4MEN01 - Introduction à l'optimisation

Programme cadre pour le TP n°2

@author: Florence Ossart, Sorbonne Université
"""

""" Dichotomie"""
        


import numpy as np 
import matplotlib.pyplot as plt

def f1(x) :
    return (x+1)**2 + 7*np.sin(x)


x_min = -4
x_max = +4
precision = 1e-1

# METHODE minimumDichotomie A CREER


def minimumDichotomie(f1,x1,x5,precision) : 
    
    intervalle  =  1
    iteration = 0 
    Xinf , Yinf = np.zeros(50) , np.zeros(50) # Tres grande dimension 
    # Ce sont des tableaux de remplissage 
    Xsup , Ysup = np.zeros(50) , np.zeros(50)
    
    F = True
    
    
    Matrix_inf , Matrix_sup = [Xinf , Yinf] , [Xsup , Ysup] # Initialisation
    
    
    while precision < abs(intervalle) :
        
        Xinf[iteration] = x1 ; Yinf[iteration] = f1(x1) # Stockage
        # chaque xmin est stocké avec son ymin ; pareil pour xmax et ymax
        Xsup[iteration] = x5 ; Ysup[iteration] = f1(x5)
        
        
        intervalle  =  x5 - x1 
        x3 = 0.5*(x1 + x5) ; x4 = 0.5*(x5+x3) ; x2 = 0.5*(x3+x1)
        
        if iteration > 20 : # Divergence de l'algo 
             F = False 
             break
        
        
        if f1(x1) < f1(x2) < f1(x3) < f1(x4) < f1(x5) : 
            """ le minimum est nécessairement dans l’intervalle [x1,x2]"""
            x5 = x2 ; iteration += 1
            continue # Pour aller au tour de boucle suivant 
        
        if f1(x1)  > f1(x2) < f1(x3) < f1(x4) < f1(x5) : 
            """ le minimum est nécessairement dans l’intervalle [x1,x3]"""
            x5 = x3 ; iteration += 1
            continue
        
        if f1(x1)  > f1(x2) > f1(x3) < f1(x4) < f1(x5) : 
            """ le minimum est nécessairement dans l’intervalle [x2,x4]"""
            x1 = x2 ; x5 = x4; iteration += 1
            continue
        
        if f1(x1)  > f1(x2) > f1(x3) > f1(x4) < f1(x5) : 
            """ le minimum est nécessairement dans l’intervalle [x3,x5]"""
            x1 = x3 ; iteration += 1
            continue
        
        if f1(x1)  > f1(x2) > f1(x3) > f1(x4) > f1(x5) : 
            """ le minimum est nécessairement dans l’intervalle [x4,x5]"""
            x1 = x4 ; iteration += 1
            continue 
        
        else : iteration += 1  ; continue
        """Cas ou l'algo diverge et on en entre dans aucun cas"""
        
    """ Vecteur contenant les bornes inf (x_inf associé a son y_inf )successives """ 
    """ Sous forme de matrice [ [xinf ] , [yinf]] """ 
    
    # Re dimensionnement 
   
   
    x_min = Xinf[0:iteration] ; y_min = Yinf[0:iteration]
    x_max = Xsup[0:iteration] ; y_max = Ysup[0:iteration]
    
    
    Matrix_inf = [x_min , y_min]
    Matrix_sup = [x_max , y_max]
    
    if F == False : return Matrix_inf , Matrix_sup , iteration , False
    return Matrix_inf , Matrix_sup , iteration , True 
    
   
bornes_min, bornes_max, n_iter, ier = minimumDichotomie(f1,x_min,x_max,precision)
        
x_min, y_min = bornes_min[0][-1], bornes_min[1][-1]
x_max, y_max = bornes_max[0][-1], bornes_max[1][-1]

# Visualisation des résultats
plt.plot(bornes_min[0],bornes_min[1],'rs', label = 'x_min')
plt.plot(bornes_max[0],bornes_max[1],'bs', label = 'x_max')
plt.legend()
plt.xlabel('Valeurs de $x$')
plt.ylabel('Valeurs de $f_1(x)$')
plt.title('Recherche du minimum de $f_1$ par dichotomie')
plt.grid()

message = 'Precision = {}'.format(precision)
message += '\nCV en {} iterations'.format(n_iter)
message += '\nBorne inférieure :'
message += '\n  x_min = {:6.4f}'.format(x_min)
message += '\n  y_min = {:6.4f}'.format(y_min)
message += '\nBorne supérieure :'
message += '\n  x_max = {:6.4f}'.format(x_max)
message += '\n  y_max = {:6.4f}'.format(y_max)
plt.text(1,-5,message)




# (x+1)^2 + 7sin(x) est unimodale , elle est dérivable et sa dérivée
#s'annule et change de signe en un unique point de l'intervalle [-4,4]

#(x+1)^2 + 10sin(x) n'est pas unimodale , le code tourne 
# en rond ,  cette methode n'est adaptée qu'aux fonctions unimodales """