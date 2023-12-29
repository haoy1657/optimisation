#%% Programme de test de la recherche de minimum par methode de Newton 
import numpy as np 
# import matplotlib.pyplot as plt
#%% Fonction test n°1
def f(x) :
    return (x+1)**2 + 7*np.sin(x)

precision = 1e-1

#%% 
def df(x) : 
  df = 2*(x+1) + 7*np.cos(x) 
  return df 

def d2f(x) : 
  d2f = 2 - 7*np.sin(x)
  return d2f 


# METHODE minimumDichotomie A CREER
#%% 
def minimumNewton(f,df,d2f,depart,precision,iterations) : 
        
# Initialisation 
 

        
        if d2f(depart) == 0 : return print('Error')
        
        Eps = 2*precision 
        
        xn = depart
        xn_1 = xn - df(xn)/d2f(xn)
        i = 0
        
      
        
        while (abs(xn_1 - xn)) > Eps and i < iterations : 
            
            if d2f(depart) == 0 : return print('Error')
            xn = xn_1
            xn_1 = xn - df(xn)/d2f(xn)
            
            i += 1 
            
   
        return (xn_1 ,i)
        

    