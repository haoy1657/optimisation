## Projet : Trajectoire d’un bras de robot à 2 articulations
# Fait par : 
# HAO Yuan 21117163 SAR
# Yifeng YU 21113616 SAR
# XUCHEN wei 28620150 SAR

# In[24]:
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# In[25]:


#Ecrire les équations du modèle direct 𝑋=F(theta), puis du résidu 𝑅(theta) pour une position d’outil donnée.
def F(theta1,theta2):#Modème directe
    X=[l1*np.cos(theta1)+l2*np.cos(theta1+theta2),l1*np.sin(theta1)+l2*np.sin(theta1+theta2)]
    return X


# In[26]:


def fun(x):#fontion de résidus
    return [l1*np.cos(x[0])  + l2*np.cos(x[0]+x[1])-P[0],
            l1*np.sin(x[0])  + l2*np.sin(x[0]+x[1])-P[1]]
def jac(x):#jacobien de résidus
    return np.array([[-l1*np.sin(x[0])-l2*np.sin(x[0]+x[1]),
                      -l2*np.sin(x[0]+x[1])],
                     [l1*np.cos(x[0])+l2*np.cos(x[0]+x[1]),
                      l2*np.cos(x[0]+x[1])]])
def f(x):#carre_norme_residus
    return (l1*np.cos(x[0])  + l2*np.cos(x[0]+x[1])-P[0])**2+(l1*np.sin(x[0])  + l2*np.sin(x[0]+x[1])-P[1])**2
def j(theta1,theta2):#jaobien de carre_norme_residus
    r1=2*(l1*np.cos(theta1)+l2*np.cos(theta1+theta2)-P[0])*(-l1*np.sin(theta1)-l2*np.sin(theta1+theta2))+2*(l1*np.sin(theta1)+l2*np.sin(theta1+theta2)-P[1])*(l1*np.cos(theta1)+l2*np.cos(theta1+theta2))
    r2=2*(l1*np.cos(theta1)+l2*np.cos(theta1+theta2)-P[0])*(-l2*np.sin(theta1+theta2))+2*(l1*np.sin(theta1)+l2*np.sin(theta1+theta2)-P[1])*(l2*np.cos(theta1+theta2))
    return r1,r2
def H(theta1,theta2):#Hessian de carre_norme_residus
    h1=2*P[0]*(l1*np.cos(theta1)+l2*np.cos(theta1+theta2))+ 2*P[1]*(l1*np.sin(theta1)+l2*np.sin(theta1+theta2))
    h2=2*P[0]*l2*np.cos(theta1+theta2)+2*P[1]*l2*np.sin(theta1+theta2)
    h3=2*P[0]*l2*np.cos(theta1+theta2)+2*P[1]*l2*np.sin(theta1+theta2)
    h4=2*(l1*np.cos(theta1)-P[0])*(-l2*np.cos(theta1+theta2))+ 2*(l1*np.sin(theta1)-P[1])*(-l2*np.sin(theta1+theta2))
    return h1,h2,h3,h4


# In[27]:


def p(angle,l):#exprimer le vecteur d'un segment avec un variable pi qui est l'anglre entre le vecteur OP et l'axe x et un variable de la module de ce vecteurm,cette fontion sera utilisée dans la fonction solution_trajectoire qui nous permet de tracer le trajectoire des deux bras d'une ou plusieur solutions
    return [l*np.cos(angle),l*np.sin(angle)]
def f1(x):#cette fonction sera utilisé dans la fonction solution_analytique,il nous petmet de calculer le coordonnée d'un intersection des deux ligne 
    return [(l1*np.cos(theta[0])-x[0])**2+(l1*np.sin(theta[0])-(P[1]/P[0])*x[0])**2-((l1*l2*np.sin(theta[1]))/(P[0]**2+P[1]**2))**2]
def jac_f1(x):#jacobien de la fontion f1
    return np.array([-2*(l1*np.cos(theta[0])-x[0])-2*(P[1]/P[0])*(l1*np.sin(theta[0])-(P[1]/P[0])*x[0])])   


# In[28]:


def gradient_pas_fix(theta,alpha,nmax):#méthode gradient a pas fixé avec trois varriable définis (theta:le point départ choisi, alpha:le pas de la méthode gradient,nmax: le nombre max d'itération)
    theta1=theta[0]
    theta2=theta[1]
    tab_theta1=[theta1]#ici ,on défini deux list, ils seront utilisés dans la fonction résultat_iteration_gradient qui permet de s'afficher le résultat d'iteration pour un point départ choisi dans la graphe
    tab_theta2=[theta2]
    f1=f([theta1,theta2])#la valeur de résidus en [theta1,theta2]
    f2=0#intialisation f2=0 et n=0
    n=0
    while (abs(f1-f2))>1e-7 and n<=nmax:
        deriv1,deriv2=j(theta1,theta2)
        theta1=theta1-alpha*deriv1
        theta2=theta2-alpha*deriv2
        tab_theta1.append(theta1)
        tab_theta2.append(theta2)
        f2=f([theta1,theta2])
        n+=1 
    return [theta1,theta2],tab_theta1,tab_theta2


# In[44]:


def Newton(theta,precision,nmax) :#méthode newton avec trois vrriables(theta:le point départ choisi,précision:la précision d'arret, nmax:le nombre max d'itération)
    global r1,r2,h1,h2,h3,h4
    theta1=theta[0]
    theta2=theta[1]
    tab_theta1=[theta1]#comme méthode gradient, on cree deux list, ils seront utulisés dans la fontion résultat_iteration_newton qui permet de s'afficher le résultat d'itetation pour un point départ choisi dans la graphe 
    tab_theta2=[theta2]
    h1,h2,h3,h4=H(theta1,theta2)
    r1,r2=j(theta1,theta2)
    dX=1#initialisation 
    n=0
    while dX >= precision and n <= nmax : 
        delta_theta1=(r2*h2-r1*h4)/(h1*h4-h2*h3)#calculer le delta_theta par la formule H_residus(theta1,theta2)*delta_theta=-J_résidus(theta1,theta2)
        delta_theta2=(r2*h1-r1*h3)/(h2*h3-h1*h4)
        theta1+=delta_theta1#renouvellement de theta1 et theta2
        theta2+=delta_theta2
        tab_theta1.append(theta1)
        tab_theta2.append(theta2)
        h1,h2,h3,h4=H(theta1,theta2)#renoubellement de la matrice hessian et jacobien
        r1,r2=j(theta1,theta2)
        dX=np.sqrt((delta_theta1)**2+(delta_theta2)**2)#renouvellement de dX
        n += 1
        
    return [theta1,theta2],tab_theta1,tab_theta2


# In[30]:


def résultat_iteration_gradient(X,l,theta,precision,nmax):#cette fonction est pour tracer le resultat d'iteration de la methode gradient a pas fixé dans le graphe 
    global P
    global l1,l2
    l1=l[0]
    l2=l[1]
    P=[X[0],X[1]]
    theta1min, theta1max, ntheta1 = -10, 10, 100 
    theta2min, theta2max, ntheta2 = -10, 10, 100
    theta1d = np.linspace(theta1min,theta1max,ntheta1)
    theta2d = np.linspace(theta2min,theta2max,ntheta2)
    Theta1d, Theta2d = np.meshgrid(theta1d, theta2d)# créer la maillage 
    nIso = 10
    plt.figure(figsize=(40, 40))
    plt.subplot(121)
    cp = plt.contour(Theta1d,Theta2d,f([Theta1d,Theta2d]),nIso)#tracer le graphe de isovaleur pour la fontion résidus
    plt.clabel(cp, inline=True,fontsize=10)
    plt.title("le résultat d'itération avec le point départ choisi par la méthode gradient a pas fixé")
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    plt.grid()
    plt.axis('square')
    plt.scatter(gradient_pas_fix(theta,precision,nmax)[1],gradient_pas_fix(theta,precision,nmax)[2])# tracer le résultat d'itération de la méthode gradient a pas fixé
    plt.plot(gradient_pas_fix(theta,precision,nmax)[1][0],gradient_pas_fix(theta,precision,nmax)[2][0],'.r',label="point départ") # Départ
    plt.plot(gradient_pas_fix(theta,precision,nmax)[1][-1],gradient_pas_fix(theta,precision,nmax)[2][-1],'.y',label="point arrivée") # Arrivée
    plt.legend()
    plt.show()


# In[31]:


def résultat_iteration_newton(X,l,theta,precision,nmax):#cette fonction est pour tracer le resultat d'iteration de la methode newton dans le graphe 
    global P
    global l1,l2
    l1=l[0]
    l2=l[1]
    P=[X[0],X[1]]
    theta1min, theta1max, ntheta1 = -10 ,10, 100   
    theta2min, theta2max, ntheta2 = -10, 10, 100
    theta1d = np.linspace(theta1min,theta1max,ntheta1)
    theta2d = np.linspace(theta2min,theta2max,ntheta2)
    Theta1d, Theta2d = np.meshgrid(theta1d, theta2d) #créer la maillage 
    nIso = 10
    plt.figure(figsize=(40, 40))
    plt.subplot(121)
    cp = plt.contour(Theta1d,Theta2d,f([Theta1d,Theta2d]),nIso)#tracer le graphe de isovaleur pour la fontion résidus
    plt.clabel(cp, inline=True,fontsize=10)
    plt.title("le résultat d'itération avec le point départ choisi par la méthode newton")
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    plt.grid()
    plt.axis('square')
    plt.scatter(Newton(theta,precision,nmax)[1],Newton(theta,precision,nmax)[2])#tracer le résultat d'itération de la méthode newton 
    plt.plot(Newton(theta,precision,nmax)[1][0],Newton(theta,precision,nmax)[2][0],'.r',label="point départ") # Départ
    plt.plot(Newton(theta,precision,nmax)[1][-1],Newton(theta,precision,nmax)[2][-1],'.y',label="point arrivée") # Arrivée
    plt.legend()
    plt.show()


# ### Première méthode : utiliser la fonction « root » de la bibliothèque scypy.optimize appliquées au résidu

# In[9]:


#méthode scipy.optimize.root
def solution_articulaire_1ere(X,l):
    #définir les varriables glaobaux
    global P #la pose donné
    global tab_x,tab_y, tab_x_prime, tab_y_prime#ici on ajouter quatre list, ils définissent les valeurs pour theta1, theta2 ,theta1_prime, theta2_prime(s'il a deux solution), ils seront utilisé pour tracer la position des deux bras a instatnt différent 
    global t#le temp défini, on défini un temps de passer la position initial a la position finale, apres le calcul, on aura une ou deux valeur pour theta,on va diviser les theta par temps, et on aura les valeur des theta a instant différent par la suite 
    global valeur_bool1,valeur_bool2# les valeurs boolennes, ils seront utilsées pour distinguer le cas pas de solution ou unique solution ou deux solution 
    global l1,l2#les longuers des deux bras
    P=[X[0],X[1]]
    l1=l[0]
    l2=l[1]
    valeur_bool1=True
    valeur_bool2=True
    theta1=[]
    theta2=[]
    t=50# temps chosi est 50
    #les point départ différent qui va etre testé pour trouver les solutions possibles, ici on choisi 49 points
    point_depart=np.array([[3,3,3,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,-2,-2,-3,-3,-3,-3,-3,-3,-3]
                          ,[-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3,-3,-2,-1,0,1,2,3]])
    #les lsits défini qui seraont utilisé dans la fonction 
    A=[]#les list A et B, on va le rempli avec theta1 filtré et theta2 filtré apres le filtrage des resultat
    B=[]
    theta=[]#le list theta va etre rempli par [theta1,theta2] apres les filtrages des résutats
    F_theta=[]#pour chaque solition trouvé([theta1,theta2]), on va calculer sa valeur f([theta1,theta2])(résidus), et on les met dans le list F_theta, ils seront utilisé dans critière de filtrage suivante
    indice=[]#le list indice nous permet de savoir de auel point de depart provient les solutions trouvées 
    indice_point_depart=0#intialisation
    print("la methode par optimize.root\n")
    print("les points départ différents pour trouver solution possible:\n49 points différent choisi:\n[3, -3], [3, -2], [3, -1], [3, 0], [3, 1], [3, 2], [3, 3], [2, -3], [2, -2], [2, -1], [2, 0],[2, 1], [2, 2], [2, 3], [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3], [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3], [-1, -3], [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3], [-2, -3], [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3], [-3, -3], [-3, -2], [-3, -1], [-3, 0], [-3, 1], [-3, 2], [-3, 3]")
    #les résultat donnée par scipy root en ces points départ different
    
    #première filtrage des résultat 
    for i in range(len(point_depart[0])):
        sol = optimize.root(fun, [point_depart[0][i], point_depart[1][i]], jac=jac, method='hybr')#pour chaque point départ, on le test avec optimize.root
        a=float('%.6f'%(float(list(sol.x)[0])%6.283185))#on met les résultat de root en format de float et avec une précision du calcul de 1e-6 et on applique la minisation des valeur articumaire aussi
        b=float('%.6f'%(float(list(sol.x)[1])%6.283185))
        theta1.append(a)#on ajoute les valeurs trouvé par root dans les list theta1 et theta2, theta1 theta2 seront s'afficher pour montrer les résultat sans filtrage 
        theta2.append(b)
        if (abs(F(a,b)[0])-sol.fun[0]-X[0])<= 1e-5 and (abs(F(a,b)[1])-sol.fun[1]-X[1])<=1e-5:#définition de la première filtrage: abs(F(a,b))-sol.fun-X=0
            A.append(a)#on ajoute le resultat qui passe le filtrage dans les list A et B
            B.append(b)
            indice.append([indice_point_depart])#on ajoute la indice point départ des résultat qui passe le filtrage dans le list indice, il sera utilisé pour touver la point départ d'une solition a la fin 
            indice_point_depart+=1
        else:
            indice_point_depart+=1
    print("\nles résultats résolu par root en 49 points départ:\ntheta1----->\n",theta1,"\ntheta2----->\n",theta2)
    print("\nEt puis on doit filtrer ces résultat,enlever les résultats équivalents")
    for i in range(len(A)):#si theta1 theta2 sont égale deux pi ou moins deux pi, il est équibalent a la angle articulaire zéro 
        if abs(A[i]-6.28)<0.1 or abs(A[i]+6.28)<0.1 or abs(A[i])<0.1:
            A[i]=0
        elif abs(B[i]-6.28)<0.1 or abs(B[i]+6.28)<0.1 or abs(B[i])<0.1:
            B[i]=0
    print("\npremière filtrage des résultat:\ntheta1----->\n",A,"\ntheta2----->\n",B)# termination de la premiere filtrage 
    # deuxièmre filtrage des résultat        
    i=0
    j=1
    while i<len(A):#on essaie de enlever les valeur équivalents 
        while j<len(A):
            if abs(A[i]-A[j])<0.1 and abs(B[i]-B[j])<0.1:
                A.pop(j)
                B.pop(j) 
                indice.pop(j)
            else:
                j+=1 
        i+=1
        j=i+1
    print("\ndeuxième filtrage des résultat:\ntheta1----->\n",A,"\ntheta2----->\n",B)#termination de la deuxième filtrage 
    for i in range(len(A)):#on ajoute les valeur resté dans le list theta et les valeur de f([theta1,theta2]) de chaque solution dans le list F_theta
        theta.append([A[i],B[i]])
    for i in range (len(theta)):
        F_theta.append(f(theta[i]))
    print("\n\n----------------------------------------------------------------------------------------------------------")
    print("\nsolution possible trouvée par les points départ différents:",theta,"\nla distace possible trouvée par les points départ différents:",F_theta)
    
    #distinguer les cas, pas de solition, unique solution ou deux solutions
    for i in range(len(theta)):#
        if f(theta[i])>1e-5:
            valeur_bool1=False
        if len(theta)==1:
            valeur_bool2=False 
    #trois cas:pas de solution, unique solutions, pas de solutions
    if valeur_bool1==False:
        print("le cas pas de solution,le point le plus proche:",theta[F_theta.index(min(F_theta))],"\net la distance le plus court:",np.sqrt(min(F_theta)))
        tab_x=np.linspace(0,theta[F_theta.index(min(F_theta))][0],t)
        tab_y=np.linspace(0,theta[F_theta.index(min(F_theta))][1],t)
        tab_x_prime=[]
        tab_y_prime=[]
    elif valeur_bool1==True and valeur_bool2==False:
        tab_x=np.linspace(0,theta[F_theta.index(min(F_theta))][0],t)
        tab_y=np.linspace(0,theta[F_theta.index(min(F_theta))][1],t)
        tab_x_prime=[]
        tab_y_prime=[]
        print("solution unique:",theta[F_theta.index(min(F_theta))],"vérification par modème directe:",F(theta[F_theta.index(min(F_theta))][0],theta[F_theta.index(min(F_theta))][1]))
    elif valeur_bool1==True and valeur_bool2==True:
        tab_x=np.linspace(0,theta[0][0],t)
        tab_y=np.linspace(0,theta[0][1],t)
        tab_x_prime=np.linspace(0,theta[1][0],t)
        tab_y_prime=np.linspace(0,theta[1][1],t)
        print("deux solutions possiblees",theta)
        print("vérification par modèle directe:")
        for i in range(len(theta)):
            print(theta[i],"--->",F(theta[i][0],theta[i][1]))
    return tab_x, tab_y, tab_x_prime, tab_y_prime, valeur_bool1, valeur_bool2


# ### Deuxième méthode : utiliser la fonction « minimize » de la bibliothèque scypy.optimize appliquées au carré de la norme du résidu.

# In[19]:


#méthode scipy.optimize.minimize 
def solution_articulaire_2eme(X,l):
    global P
    global l1,l2
    l1=l[0]#les longueurs des deux bras 
    l2=l[1]
    P=[X[0],X[1]]#la pose donné
    A=[]#les list A et B, on va le rempli avec theta1 filtré et theta2 filtré apres le filtrage des resultat
    B=[]
    theta=[]#le list theta va etre rempli par [theta1,theta2] apres les filtrages des résutats
    theta1=[]
    theta2=[]
    F_theta=[]
    indice=[]#le list indice nous permet de savoir de auel point de depart provient les solutions trouvées 
    indice_point_depart=0#intialisation
    valeur_bool1=True# les valeurs boolennes, ils seront utilsées pour distinguer le cas pas de solution ou unique solution ou deux solution
    valeur_bool2=True
    #les point départ différent qui va etre testé pour trouver les solutions possibles, ici on choisi 100 points
    point_depart=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 0.6981317007977318, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 1.3962634015954636, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 2.792526803190927, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 3.490658503988659, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.1887902047863905, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 4.886921905584122, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 5.585053606381854, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586]
                          ,[0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586, 0.0, 0.6981317007977318, 1.3962634015954636, 2.0943951023931953, 2.792526803190927, 3.490658503988659, 4.1887902047863905, 4.886921905584122, 5.585053606381854, 6.283185307179586]])
    
    #les résultat donnée par scipy minimize en ces points départ different
    #première filtrage des résultat 
    for i in range(len(point_depart[0])):
        sol = optimize.minimize(f, [point_depart[0][i], point_depart[1][i]])#pour chaque point départ, on le test avec optimize.minimize
        a=float('%.6f'%(float(list(sol.x)[0])%6.283185))#on met les résultat de root en format de float et avec une précision du calcul de 1e-6 et on applique la minisation des valeur articumaire aussi
        b=float('%.6f'%(float(list(sol.x)[1])%6.283185))
        theta1.append(a)#on ajoute les valeurs trouvé par minimize dans les list theta1 et theta2, theta1 theta2 seront s'afficher pour montrer les résultat sans filtrage 
        theta2.append(b)
        #définition de la première filtrage: pour l1>l2:abs(np.sqrt((F(a,b)[0])**2+(F(a,b)[1])**2)+np.sqrt(abs(sol.fun))-np.sqrt(X[0]**2+X[1]**2))=0
        #pour l1<l2:abs(np.sqrt((F(a,b)[0])**2+(F(a,b)[1])**2)-np.sqrt(abs(sol.fun))-np.sqrt(X[0]**2+X[1]**2))
        if (abs(np.sqrt((F(a,b)[0])**2+(F(a,b)[1])**2)+np.sqrt(abs(sol.fun))-np.sqrt(X[0]**2+X[1]**2)))<= 1e-5 or (abs(np.sqrt((F(a,b)[0])**2+(F(a,b)[1])**2)-np.sqrt(abs(sol.fun))-np.sqrt(X[0]**2+X[1]**2)))<= 1e-5 :
            A.append(a)#on ajoute le resultat qui passe le filtrage dans les list A et B
            B.append(b)
            indice.append([indice_point_depart])#on ajoute la indice point départ des résultat qui passe le filtrage dans le list indice, il sera utilisé pour touver la point départ d'une solition a la fin 
            indice_point_depart+=1
        else:
            indice_point_depart+=1
    print("la methode par optimize.minimize\n")
    print("\nles résultats résolu par minimize en 100 points départ:\ntheta1----->\n",theta1,"\ntheta2----->\n",theta2)
    print("\nEt puis on doit filtrer ces résultat,enlever les résultats équivalents")
     
    for i in range(len(A)):#si theta1 theta2 sont égale deux pi ou moins deux pi, il est équibalent a la angle articulaire zéro
        if abs(A[i]-6.28)<0.1 or abs(A[i]+6.28)<0.1 or abs(A[i])<0.1:
            A[i]=0
        elif abs(B[i]-6.28)<0.1 or abs(B[i]+6.28)<0.1 or abs(B[i])<0.1:
            B[i]=0
    print("\npremière filtrage des résultat:\ntheta1----->\n",A,"\ntheta2----->\n",B)# termination de la premiere filtrage 
    # deuxièmre filtrage des résultat        
    i=0
    j=1
    while i<len(A):#on essaie de enlever les valeur équivalents 
        while j<len(A):
            if abs(A[i]-A[j])<0.1 and abs(B[i]-B[j])<0.1:
                A.pop(j)
                B.pop(j) 
                indice.pop(j)
            else:
                j+=1 
        i+=1
        j=i+1
    print("\ndeuxième filtrage des résultat:\ntheta1----->\n",A,"\ntheta2----->\n",B)#termination de la deuxième filtrage 
    for i in range(len(A)):#on ajoute les valeur resté dans le list theta et les valeur de f([theta1,theta2]) de chaque solution dans le list F_theta
        theta.append([A[i],B[i]])
    for i in range (len(theta)):
        F_theta.append(f(theta[i]))
    print("\n\n----------------------------------------------------------------------------------------------------------")
    print("\nsolution possible trouvée par les points départ différents:",theta,"\nla distace possible trouvée par les points départ différents:",F_theta)
    
    #distinguet les cas, pas de solition, unique solution ou deux solutions
    for i in range(len(theta)):
        if f(theta[i])>0.1:
            valeur_bool1=False
        if len(theta)==1:
            valeur_bool2=False
    #trois cas:pas de solution, unique solutions, pas de solutions
    if valeur_bool1==False:
        print("le cas pas de solution,le point le plus proche:",theta[F_theta.index(min(F_theta))],"\net la distance le plus court:",np.sqrt(min(F_theta)))
    elif valeur_bool1==True and valeur_bool2==False:
        print("solution unique:",theta[F_theta.index(min(F_theta))],"vérification par modème directe:",F(theta[F_theta.index(min(F_theta))][0],theta[F_theta.index(min(F_theta))][1]))
    elif valeur_bool1==True and valeur_bool2==True:
        print("deux solutions possiblees",theta)
        print("vérification par modèle directe:")
        for i in range(len(theta)):
            print("point depart(",[float(point_depart[0][indice[i]]),float(point_depart[1][indice[i]])],")   --->",theta[i],"--->",F(theta[i][0],theta[i][1]))
    print("\ncomment faut-il interpréter un minimum qui ne serait pas nul ?\nsi le minimum n'est pas nul, ca veux dire,le position de P est hors de portée du bras robotique")
    return


# ### Troisième méthode : écrire le programme pour minimiser le carré de la norme du résidu en appliquant la méthode du gradient à pas fixe, avec un mécanisme qui garantit que la norme du résidu diminue à chaque itération

# In[20]:


#méthode grandient a pas fix 
def solution_articulaire_3eme(X,l,alpha,nmax):
    #définir les varriables glaobaux
    global P#la pose donné
    global l1,l2#les longuers des deux bras
    global Alpha#le pas de la méthode gradient
    global Nmax
    Alpha=alpha#choix du pas 
    Nmax=nmax#choix de la nombre d'itération max
    l1=l[0]
    l2=l[1]
    P=[X[0],X[1]]
    indice=[]
    theta=[]#le list theta va etre rempli par [theta1,theta2] d'apres les résutats
    F_theta=[]#pour chaque solition trouvé([theta1,theta2]), on va calculer sa valeur f([theta1,theta2])(résidus), et on les met dans le list F_theta, ils seront utilisé dans critière de filtrage suivante
    valeur_bool1=True# la valeur booleene, il sera utilsée pour distinguer les cas l1<=l2 ou l1>l2, si l1<=l2-->l'espace atteignable est composé par une cercle
                     #si l1>l2--->l'espace atteignable est composé par l'espace entre le cercle extérieur et le cercle intérieur
    indice_point_depart=0#intialisation
    #les point départ différent qui va etre testé pour trouver les solutions possibles, ici on choisi 400 points
    point_depart=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586] 
                                 ,[0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586]])
    
    print("la méthode de gradient a pas fixé avec alpha choisi--->",Alpha," et le nombre d'itération max--->",Nmax)
    print("\nveuillez patienter, les donneés en train de chargement, la compilation peut prendre jusqu'a trois minute")
    if l2<l1:valeur_bool1=False
    if l2>=l1:valeur_boo1=True
    for i in range(len(point_depart[0])):
        sol = gradient_pas_fix([point_depart[0][i],point_depart[1][i]],Alpha,Nmax)[0]#pour chaque point départ, on le test avec la méthode de newton
        a=float('%.6f'%(sol[0]%6.283185))#on met les résultat de gradient en format de float et avec une précision du calcul de 1e-6 et on applique la minisation des valeur articumaire aussi
        b=float('%.6f'%(sol[1]%6.283185))
        indice.append([indice_point_depart])#on ajoute la indice de point départ des résultat  dans le list indice, il sera utilisé pour touver la point départ d'une solition a la fin 
        indice_point_depart+=1
        theta.append([a,b])#on ajoute les résultat de gradient dans un list 
        F_theta.append(np.sqrt(f([a,b])))#on ajoute le raccine de residus pour chaque point départ dans un list 
    for i in range (len(F_theta)-1):#Nous trions les résultats de la valeur de residus  du plus petit au plus grand 
        for j in range (len(F_theta)-1-i):
            if F_theta[j]>F_theta[j+1]:
                F_theta[j],F_theta[j+1]=F_theta[j+1],F_theta[j]
                theta[j],theta[j+1]=theta[j+1],theta[j]
                indice[j],indice[j+1]=indice[j+1],indice[j]
    choix_affiche=input("veuillez taper 'yes' si vous voulez afficher les resultats de methode gradient a pas fix sinon taper'no'--->")
    if choix_affiche=="yes":
        print("\nles résultats résolu par methode gradient a pas fixe en 400 points départ:\n\ntheta----->\n",theta,"\n\nracinne du residus possible----->\n",F_theta)
    print("\npour la pose--->",P,"et les longuer des bras--->",l,"  les resultats d'analyse finale par methode gradient sont:")
    if F_theta[0]>=1e-4:#le valeur minimum de residus obtenur par gradient est trop grand en ce cas, ca veux dire il n'y a pas de solution, on prend racinne de la valeur minimum de residu comme la distanc le plus court, et le theta correspondant comme la solution trouvée
        print("\nle cas pas de solution,le point le plus proche:","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\net la distance le plus court:",F_theta[0])
        résultat_iteration_gradient(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Alpha,Nmax)#tracer le resultat d'iteration pour la solution trouvée dans un graphe
    elif valeur_bool1==True and (P[0]**2+P[1]**2)==(l1+l2)**2:#solution unique, ici, comme le cas pas de solution, on prends la valeur min de la racinne du residus, et le theta correspondant comme solution trouvée
        print("\nle cas solution unique,","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\nvérification par modele geometrique:",F(theta[0][0],theta[0][1]))
        résultat_iteration_gradient(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Alpha,Nmax)
    elif valeur_bool1==False and ((P[0]**2+P[1]**2)==(l1+l2)**2 or (P[0]**2+P[1]**2)==(l1-l2)**2):
        print("\nle cas unique solution,","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\nvérification par modele geometrique:",F(theta[0][0],theta[0][1]))
        résultat_iteration_gradient(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Alpha,Nmax)
    else:#le cas pas de solution, on sait bien il y a deux soltions possibles dans les resultat trouvés par gradient en 400 point départ, on doit filtrer ces resultat, enlever les resultat équivalent, et on va trouver deux solution différent trouvées par gradient
        for i in range(len(theta)):#si theta1 theta2 sont égale deux pi ou moins deux pi, il est équibalent a la angle articulaire zéro
            if abs(theta[i][0]-6.28)<0.1 or abs(theta[i][0]+6.28)<0.1 or abs(theta[i][0])<0.1:
                theta[i][0]=0
            elif abs(theta[i][1]-6.28)<0.1 or abs(theta[i][1]+6.28)<0.1 or abs(theta[i][1])<0.1:
                theta[i][1]=0
        i=0
        j=1
        while i<len(theta):#on essaie de enlever les valeur équivalents
            while j<len(theta):
                if abs(theta[i][0]-theta[j][0])<0.1 and abs(theta[i][1]-theta[j][1])<0.1:
                    theta.pop(j)
                    F_theta.pop(j)
                    indice.pop(j)
                else:
                    j+=1 
            i+=1
            j=i+1 
        print("\napres de le filtrage des valeurs équivalents\ntheta---->\n",theta,"\nla valeur de racinne de residus--->",F_theta)
        #on prends les deux première solutions apres le filtrage comme les deux solutions finales
        for i in range(len(theta)-2):
                theta.pop(2)
                F_theta.pop(2)
                indice.pop(2)
        print("\nles solutions posisbles trouvées--->",theta,"\n")
        print("solution non_uniques!")
        print("premiere solution nous donne--->","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")--->",theta[0],"  vérification par modèle geometrique--->",F(theta[0][0],theta[0][1]))
        résultat_iteration_gradient(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Alpha,Nmax)
        print("deuxieme solution nous donne--->","point depart(",[float(point_depart[0][indice[1]]),float(point_depart[1][indice[1]])],")--->",theta[1],"  vérification par modèle geometrique--->",F(theta[1][0],theta[1][1]))
        résultat_iteration_gradient(X,l,[float(point_depart[0][indice[1]]),float(point_depart[1][indice[1]])],Alpha,Nmax)     
    return 


# ### Quatrième méthode : écrire le programme pour minimiser le carré de la norme du résidu en appliquant la méthode de Newton.

# In[47]:


def solution_articulaire_4eme(X,l,precision,nmax):
    #définir les varriables glaobaux
    global P#la pose donné
    global l1,l2#les longuers des deux bras
    global Precision#le précision de la méthode newton
    global Nmax
    Precision=precision#choix de la précion 
    Nmax=nmax#choix du nombre max d'itération 
    l1=l[0]
    l2=l[1]
    P=[X[0],X[1]]
    indice=[]
    theta=[]#le list theta va etre rempli par [theta1,theta2] d'apres les résutats
    F_theta=[]#pour chaque solition trouvé([theta1,theta2]), on va calculer sa valeur f([theta1,theta2])(résidus), et on les met dans le list F_theta, ils seront utilisé dans critière de filtrage suivante
    valeur_bool1=True# la valeur booleene, il sera utilsée pour distinguer les cas l1<=l2 ou l1>l2, si l1<=l2-->l'espace atteignable est composé par une cercle
                     #si l1>l2--->l'espace atteignable est composé par l'espace entre le cercle extérieur et le cercle intérieur
    indice_point_depart=0
    #les point départ différent qui va etre testé pour trouver les solutions possibles, ici on choisi 400 points
    point_depart=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.3306939635357677, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.6613879270715354, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 0.992081890607303, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.3227758541430708, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.6534698176788385, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 1.984163781214606, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.3148577447503738, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.6455517082861415, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 2.9762456718219092, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.306939635357677, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.6376335988934447, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 3.968327562429212, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.29902152596498, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.6297154895007475, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 4.960409453036515, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.291103416572283, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.621797380108051, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 5.9524913436438185, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586, 6.283185307179586]
    ,[0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586, 0.0, 0.3306939635357677, 0.6613879270715354, 0.992081890607303, 1.3227758541430708, 1.6534698176788385, 1.984163781214606, 2.3148577447503738, 2.6455517082861415, 2.9762456718219092, 3.306939635357677, 3.6376335988934447, 3.968327562429212, 4.29902152596498, 4.6297154895007475, 4.960409453036515, 5.291103416572283, 5.621797380108051, 5.9524913436438185, 6.283185307179586]])
        
    print("la methode de newton avec la précision choisi--->",Precision,"et le nombre d'itération max--->",Nmax)
    print("\nveuillez patienter, les donneés en train de chargement, la compilation peut prendre jusqu'a trois minute")
    if l2<l1:valeur_bool1=False
    if l2>=l1:valeur_boo1=True
    for i in range(len(point_depart[0])):
        sol = Newton([point_depart[0][i],point_depart[1][i]],Precision,Nmax)[0]#pour chaque point départ, on le test avec la méthode de newton
        a=float('%.6f'%(sol[0]%6.283185))#on met les résultat de root en format de float et avec une précision du calcul de 1e-6 et on applique la minisation des valeur articumaire aussi
        b=float('%.6f'%(sol[1]%6.283185))
        indice.append([indice_point_depart])#on ajoute la indice point départ des résultat dans le list indice, il sera utilisé pour touver la point départ d'une solition a la fin 
        indice_point_depart+=1
        theta.append([a,b])#on ajoute les résultat de gradient dans un list 
        F_theta.append(np.sqrt(f([a,b])))#on ajoute le raccine de residus pour chaque point départ dans un list
    for i in range (len(F_theta)-1):#Nous trions les résultats de la valeur de residus  du plus petit au plus grand 
        for j in range (len(F_theta)-1-i):
            if F_theta[j]>F_theta[j+1]:
                F_theta[j],F_theta[j+1]=F_theta[j+1],F_theta[j]
                theta[j],theta[j+1]=theta[j+1],theta[j]
                indice[j],indice[j+1]=indice[j+1],indice[j]
    choix_affiche=input("veuillez taper 'yes' si vous voulez afficher les resultats de methode newton sinon taper 'no'--->")
    if choix_affiche=="yes":
        print("\nles résultats résolu par methode newton en 400 points départ:\n\ntheta----->\n",theta,"\n\nracinne du residus possible----->\n",F_theta)
    print("\n\npour la pose--->",P,"et les longuer des bras--->",l,"  les resultats d'analyse finale par methode gradient sont:")
    if F_theta[0]>=1e-4:#le valeur minimum de residus obtenur par newton est trop grand en ce cas, ca veux dire il n'y a pas de solution, on prend racinne de la valeur minimum de residu comme la distanc le plus court, et le theta correspondant comme la solution trouvée
        print("\nle cas pas de solution,le point le plus proche:","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\net la distance le plus court:",F_theta[0])
        résultat_iteration_newton(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Precision,Nmax)#tracer le resultat d'iteration pour la solution trouvée dans un graphe
        point_depart=[[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])]]
    elif valeur_bool1==True and (P[0]**2+P[1]**2)==(l1+l2)**2:#solution unique, ici, comme le cas pas de solution, on prends la valeur min de la racinne du residus, et le theta correspondant comme solution trouvée
        print("\nle cas solution unique,","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\nvérification par modele geometrique:",F(theta[0][0],theta[0][1]))
        résultat_iteration_newton(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Precision,Nmax)
        point_depart=[[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])]]
    elif valeur_bool1==False and ((P[0]**2+P[1]**2)==(l1+l2)**2 or (P[0]**2+P[1]**2)==(l1-l2)**2):
        print("\nle cas unique solution,","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")   --->theta:",theta[0],"\nvérification par modele geometrique:",F(theta[0][0],theta[0][1]))
        résultat_iteration_newton(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Precision,Nmax)
        point_depart=[[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])]]
    else:#le cas pas de solution, on sait bien il y a deux soltions possibles dans les resultat trouvés par newton en 400 point départ, on doit filtrer ces resultat, enlever les resultat équivalent, et on va trouver deux solution différent trouvées par gradient
        for i in range(len(theta)):#si theta1 theta2 sont égale deux pi ou moins deux pi, il est équibalent a la angle articulaire zéro
            if abs(theta[i][0]-6.28)<0.1 or abs(theta[i][0]+6.28)<0.1 or abs(theta[i][0])<0.1:
                theta[i][0]=0
            elif abs(theta[i][1]-6.28)<0.1 or abs(theta[i][1]+6.28)<0.1 or abs(theta[i][1])<0.1:
                theta[i][1]=0
        i=0
        j=1
        while i<len(theta):#on essaie de enlever les valeur équivalents 
            while j<len(theta):
                if abs(theta[i][0]-theta[j][0])<0.1 and abs(theta[i][1]-theta[j][1])<0.1:
                    theta.pop(j)
                    F_theta.pop(j)
                    indice.pop(j)
                else:
                    j+=1 
            i+=1
            j=i+1 
        print("\napres de le filtrage des valeurs équivalents---->\n",theta)
        #on prends les deux première solutions apres le filtrage comme les deux solutions finales
        for i in range(len(theta)-2):
                theta.pop(2)
                F_theta.pop(2)
                indice.pop(2)
        print("\nles solutions posisbles trouvées--->",theta,"\n")
        print("solution non_uniques!")
        print("premiere solution nous donne--->","point depart(",[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],")--->",theta[0],"  vérification par modèle geometrique--->",F(theta[0][0],theta[0][1]))
        résultat_iteration_newton(X,l,[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],Precision,Nmax)
        print("deuxieme solution nous donne--->","point depart(",[float(point_depart[0][indice[1]]),float(point_depart[1][indice[1]])],")--->",theta[1],"  vérification par modèle geometrique--->",F(theta[1][0],theta[1][1]))
        résultat_iteration_newton(X,l,[float(point_depart[0][indice[1]]),float(point_depart[1][indice[1]])],Precision,Nmax)  
        point_depart=[[float(point_depart[0][indice[0]]),float(point_depart[1][indice[0]])],[float(point_depart[0][indice[1]]),float(point_depart[1][indice[1]])]]
        
    return point_depart



# In[ ]:


#tracer la position du bras articulé à différents instants,ici on prends le résultat obtenu par la méthode optimize root 
def corrdonnee_point(theta1,theta2):#cette sera utilisée dans la fontion solution_et_trajectoire
    return [[l1*np.cos(theta1),l1*np.sin(theta1)],[l1*np.cos(theta1)+l2*np.cos(theta1+theta2),l1*np.sin(theta1)+l2*np.sin(theta1+theta2)]]
def cercle1(x):#définir l'espace atteignable
    return np.sqrt((l1+l2)**2-x**2)
def cercle2(x):
    return -np.sqrt((l1+l2)**2-x**2)
get_ipython().run_line_magic('matplotlib', 'qt5')
def solution_et_trajectoire(X,l):
    #le temps t est défini dans la fontion solution_articulaire_1ere----50
    l1=l[0]
    l2=l[1]
    solution_articulaire_1ere(X,l)
    plt.figure(figsize=(10, 10))
    plt.ion()
    if valeur_bool1==True and valeur_bool2==False :#unique solution
        for i in range(t):
            tabX = np.linspace(-l1-l2,l1+l2,2000)
            tabY = cercle1(tabX)
            tabY1= cercle2(tabX)
            plt.cla() #enlever l'inamge derniere
            plt.title("la position du bras articulé à différents instants avec le temps:10secondes\nunique solution possible")
            plt.grid()
            plt.xlabel("x")
            plt.xlim(-l1-l2-1, l1+l2+1)
            plt.xticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
            plt.ylabel("y")
            plt.ylim(-l1-l2-1, l1+l2+1)
            plt.yticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
        
            plt.plot([0,corrdonnee_point(tab_x[i],tab_y[i])[0][0]],[0, corrdonnee_point(tab_x[i],tab_y[i])[0][1]], c='r')
            plt.plot([corrdonnee_point(tab_x[i],tab_y[i])[0][0],corrdonnee_point(tab_x[i],0)[1][0]],[ corrdonnee_point(tab_x[i],tab_y[i])[0][1],corrdonnee_point(tab_x[i],0)[1][1]], c='r',label="solution1")
            plt.plot(corrdonnee_point(tab_x[i],0)[0][0],corrdonnee_point(tab_x[i],0)[0][1],'o',color='k',label="pivot Rotoide")
            plt.plot(P[0],P[1],'o',color='y',label="la pose donnée")
            plt.plot(tabX, tabY,'k-',label="l'espace atteignable")
            plt.plot(tabX, tabY1,'k-')
            plt.legend(loc='upper right',prop = {'size':11})
            plt.pause(0.1)
        
    
    if valeur_bool1==True and valeur_bool2==True : #deux solutions possibles 
        for i in range(t):
            tabX = np.linspace(-l1-l2,l1+l2,2000)
            tabY = cercle1(tabX)
            tabY1= cercle2(tabX)
            plt.cla() #enlever l'inamge derniere
            plt.title("la position du bras articulé à différents instants avec le temps:10secondes\ndeux solutions possibles")
            plt.grid()
            plt.xlabel("x")
            plt.xlim(-l1-l2-1, l1+l2+1)
            plt.xticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
            plt.ylabel("y")
            plt.ylim(-l1-l2-1, l1+l2+1)
            plt.yticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
        
            plt.plot([0,corrdonnee_point(tab_x[i],tab_y[i])[0][0]],[0, corrdonnee_point(tab_x[i],tab_y[i])[0][1]], c='r')#trance le segement l1
            plt.plot([corrdonnee_point(tab_x[i],tab_y[i])[0][0],corrdonnee_point(tab_x[i],tab_y[i])[1][0]],[ corrdonnee_point(tab_x[i],tab_y[i])[0][1],corrdonnee_point(tab_x[i],tab_y[i])[1][1]], c='r',label="solution1")#trance le segement l2
            
            plt.plot([0,corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][0]],[0, corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][1]], c='b')#trance le segement l1 pour la solution2
            plt.plot([corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][0],corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[1][0]],[ corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][1],corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[1][1]], c='b',label="solution2")#trance le segement l2 pour la solution2
            
            plt.plot(corrdonnee_point(tab_x[i],0)[0][0],corrdonnee_point(tab_x[i],0)[0][1],'o',color='k',label="pivot Rotoide")#tracer le pivot rotoide pour la solition1
            plt.plot(corrdonnee_point(tab_x_prime[i],0)[0][0],corrdonnee_point(tab_x_prime[i],0)[0][1],'o',color='k')#tracer le pivot rotoide pour la soliution2
            
            plt.plot(P[0],P[1],'o',color='y',label="la pose donnée")
            plt.plot(tabX, tabY,'k-',label="l'espace atteignable")#tracer l'espace atteignable
            plt.plot(tabX, tabY1,'k-')
            plt.legend(loc='upper right',prop = {'size':11})
            plt.pause(0.1)
        
      
    
    if valeur_bool1==False  :#pas de solution solution
        for i in range(t):
            
            tabX = np.linspace(-l1-l2,l1+l2,2000)
            tabY = cercle1(tabX)
            tabY1= cercle2(tabX)
            plt.cla() #enlever l'inamge derniere
            plt.title("la position du bras articulé à différents instants avec le temps:10secondes\nla pose donnée est hors la porté atteignable")
            plt.grid()
            plt.xlabel("x")
            plt.xlim(-l1-l2-1, l1+l2+1)
            plt.xticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
            plt.ylabel("y")
            plt.ylim(-l1-l2-1, l1+l2+1)
            plt.yticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
        
            plt.plot([0,corrdonnee_point(tab_x[i],tab_y[i])[0][0]],[0, corrdonnee_point(tab_x[i],tab_y[i])[0][1]], c='r')
            plt.plot([corrdonnee_point(tab_x[i],tab_y[i])[0][0],corrdonnee_point(tab_x[i],0)[1][0]],[ corrdonnee_point(tab_x[i],tab_y[i])[0][1],corrdonnee_point(tab_x[i],0)[1][1]], c='r',label="solution1") 
            plt.plot(corrdonnee_point(tab_x[i],0)[0][0],corrdonnee_point(tab_x[i],0)[0][1],'o',color='k',label="pivot Rotoide")
            plt.plot(P[0],P[1],'o',color='y',label="la pose donnée")
            plt.plot(tabX, tabY,'k-',label="l'espace atteignable")#tracer l'espace atteignable
            plt.plot(tabX, tabY1,'k-')
            plt.legend(loc='upper right',prop = {'size':11})
            plt.pause(0.1)
        
       
    
    plt.ioff()
    plt.show()
    return

#tracer en utilisant la méthode newton
def solution_et_trajectoire_avec_resultat_iteration_newton(X,l,precision,nmax):
    #le temps t est défini dans la fontion solution_articulaire_1ere----50
    l1=l[0]
    l2=l[1]
    point_depart=solution_articulaire_4eme(X,l,precision,nmax)#en donnant les parametre X,L,alpha,nmax,solution_articulaire_3eme va nous donner une ou deux point points départ, on va utiliser ces point pour refaire l'iteration du gradient
    
    plt.figure(figsize=(10, 10))
    plt.ion()
    if len(point_depart)==1 :#unique solution ou pas de solution
        [theta1,theta2],tab_x,tab_y=Newton(point_depart[0],precision,nmax)
        for i in range(len(tab_x)):
            tabX = np.linspace(-l1-l2,l1+l2,2000)
            tabY = cercle1(tabX)
            tabY1= cercle2(tabX)
            plt.cla() #enlever l'inamge derniere
            plt.title("la position du bras articulé à différents instants d'apres le resultat d'iteration de la méthode newton")
            plt.grid()
            plt.xlabel("x")
            plt.xlim(-l1-l2-1, l1+l2+1)
            plt.xticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
            plt.ylabel("y")
            plt.ylim(-l1-l2-1, l1+l2+1)
            plt.yticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
        
            plt.plot([0,corrdonnee_point(tab_x[i],tab_y[i])[0][0]],[0, corrdonnee_point(tab_x[i],tab_y[i])[0][1]], c='r')#tracer segement l1 pour la solution1
            plt.plot([corrdonnee_point(tab_x[i],tab_y[i])[0][0],corrdonnee_point(tab_x[i],0)[1][0]],[ corrdonnee_point(tab_x[i],tab_y[i])[0][1],corrdonnee_point(tab_x[i],0)[1][1]], c='r',label="solution unique ou solution plus proche")#tracer segement l2 pour la solution2
            plt.plot(corrdonnee_point(tab_x[i],0)[0][0],corrdonnee_point(tab_x[i],0)[0][1],'o',color='k',label="pivot Rotoide")#tracer le pivot totoide
            plt.plot(P[0],P[1],'o',color='y',label="la pose donnée")
            plt.plot(tabX, tabY,'k-',label="l'espace atteignable")
            plt.plot(tabX, tabY1,'k-')
            plt.legend(loc='upper right',prop = {'size':11})
            plt.pause(0.001)
        
    
    if len(point_depart)==2 : #deux solutions possibles 
        
        [theta1,theta2],tab_x,tab_y=Newton(point_depart[0],precision,nmax)
        [theta1_prime,theta2_prime],tab_x_prime,tab_y_prime=Newton(point_depart[1],precision,nmax)
        for i in range(min(len(tab_x),len(tab_x_prime))):
            tabX = np.linspace(-l1-l2,l1+l2,2000)
            tabY = cercle1(tabX)
            tabY1= cercle2(tabX)
            plt.cla() #enlever l'inamge derniere
            plt.title("la position du bras articulé à différents instants d'apres le resultat d'iteration de la méthode newton")
            plt.grid()
            plt.xlabel("x")
            plt.xlim(-l1-l2-1, l1+l2+1)
            plt.xticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
            plt.ylabel("y")
            plt.ylim(-l1-l2-1, l1+l2+1)
            plt.yticks(np.linspace(-l1-l2-5, l1+l2+5, 15, endpoint=True))
        
            plt.plot([0,corrdonnee_point(tab_x[i],tab_y[i])[0][0]],[0, corrdonnee_point(tab_x[i],tab_y[i])[0][1]], c='r')
            plt.plot([corrdonnee_point(tab_x[i],tab_y[i])[0][0],corrdonnee_point(tab_x[i],tab_y[i])[1][0]],[ corrdonnee_point(tab_x[i],tab_y[i])[0][1],corrdonnee_point(tab_x[i],tab_y[i])[1][1]], c='r',label="solution1")
            
            plt.plot([0,corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][0]],[0, corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][1]], c='b')
            plt.plot([corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][0],corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[1][0]],[ corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[0][1],corrdonnee_point(tab_x_prime[i],tab_y_prime[i])[1][1]], c='b',label="solution2")
            
            plt.plot(corrdonnee_point(tab_x[i],0)[0][0],corrdonnee_point(tab_x[i],0)[0][1],'o',color='k',label="pivot Rotoide")
            plt.plot(corrdonnee_point(tab_x_prime[i],0)[0][0],corrdonnee_point(tab_x_prime[i],0)[0][1],'o',color='k')
            
            plt.plot(P[0],P[1],'o',color='y',label="la pose donnée")
            plt.plot(tabX, tabY,'k-',label="l'espace atteignable")#tracer l'espace atteignable
            plt.plot(tabX, tabY1,'k-')
            plt.legend(loc='upper right',prop = {'size':11})
            plt.pause(0.001)
       
    
    plt.ioff()
    plt.show()
    return

# In[ ]:


#méthode analytique:on utilise la calcul mathématique pour trouver une soltion deux solution ou pas de solution
def solution_articulaire_analytique(X,l):
    module_vecteurP=X[0]**2+X[1]**2 #Px au carré plus Py au carré
    global P
    global theta,theta_prime
    global l1,l2
    l1=l[0]
    l2=l[1]
    P=[X[0],X[1]]
    #deux cas globaux, l1>=l2 ou l1<l2
    print("la methode anamytique\n")
    if l2>l1:
        if module_vecteurP==(l1+l2)**2:#le cas unique solution
            sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
            theta0=sol.x
            theta=[theta0[0]%(2*np.pi),theta0[1]%(2*np.pi)]#minimiser le valeur articulaire
            theta_prime="il y a q'une solution"
            a=F(theta[0],theta[1])
            print(' solution unique:',theta)
            print("\n vérification de la valeur\n pour theta,modèle directe nous donne:",a)
        elif module_vecteurP<(l1+l2)**2: #le cas deux solutions
            #première soluiton 
            sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
            theta0=sol.x
            theta=[theta0[0]%(2*np.pi),theta0[1]%(2*np.pi)]#%(2*np.pi)
            #deuxième soluiton possible donnée par la methode analytique, apres d'avoir trouvé une solution, on va utiliser le formule mathematique pour calculer une autre solution directement
            sol1=optimize.root(f1, [0], jac=jac_f1, method='hybr')
            P_B_x=sol1.x
            #en génerale, on utilise le symétrie pour trouver une autre solution,les deux solition sont symétrique par rappor a la ligne OP(O est le point origine et P est le point de la pose)
            P_A=p(theta[0],l1)
            P_B=[P_B_x,(X[1]/X[0])*P_B_x]
            vecteur_AC=[2*(P_B[0]-P_A[0]),2*(P_B[1]-P_A[1])]
            P_C=[vecteur_AC[0]+P_A[0],vecteur_AC[1]+P_A[1]]
            vec_CE=[X[0]-P_C[0],X[1]-P_C[1]]
            theta_prime=[float(np.arctan2(P_C[1],P_C[0])%(2*np.pi)),float((np.arctan2(vec_CE[1],vec_CE[0])-np.arctan2(P_C[1],P_C[0]))%(2*np.pi))]
            a=F(theta[0],theta[1])
            b=F(theta_prime[0],theta_prime[1])
            print(" solution non-uniques:\n la première solution:",theta,"\n","l'autre solution :",theta_prime)
            print("\n vérification de la valeur:\n pour theta,modèle directe nous donne:",a,"\n","pour theta_prime,modèle directe nous donne:",b)
            print("en génerale, on utilise le symétrie pour trouver une autre solution,les deux solition sont symétrique par rappor a la ligne OP(O est le point origine et P est le point de la pose)")
        else:
            #pas de solution 
            sol=optimize.minimize(f, [1, 1])#[1,1]est le point départ pour minimiser la fonction
            print("la pose donnée est hors la porté atteignable")
            print("le point le plus proche de la pose donnée:",sol.x)
            print("la distance entre la pose donné et le point qui est le plus proche de la pose donnée:",np.sqrt(abs(sol.fun)))
            theta="il n'y pas de solution pour ce cas,hors de portée du bras robotique"
            theta_prime="il n'y pas de solution pour ce cas,hors de portée du bras robotique"  
    else:
        if module_vecteurP==(l1+l2)**2 or module_vecteurP==(l1-l2)**2: #solution unique, on utilise la première méthode 'root' pour le résoudre  
            sol = optimize.root(fun, [0, 0], jac=jac, method='hybr')
            theta0=sol.x
            theta=[theta0[0]%(2*np.pi),theta0[1]%(2*np.pi)]
            theta_prime="il y a q'une solution"
            a=F(theta[0],theta[1])
            print(' solution unique:',theta)
            print("\n vérification de la valeur\n pour theta,modèle directe nous donne:",a)   
        elif module_vecteurP<(l1+l2)**2 and module_vecteurP>(l1-l2)**2:#deux solutions possibles 
            #première soluiton 
            sol = optimize.root(fun, [0, 1], jac=jac, method='hybr')
            theta0=sol.x
            theta=[theta0[0]%(2*np.pi),theta0[1]%(2*np.pi)]#%(2*np.pi)
            #deuxième soluiton possible
            sol1=optimize.root(f1, [0], jac=jac_f1, method='hybr')
            P_B_x=sol1.x
            P_A=p(theta[0],l1)
            P_B=[P_B_x,(X[1]/X[0])*P_B_x]
            vecteur_AC=[2*(P_B[0]-P_A[0]),2*(P_B[1]-P_A[1])]
            P_C=[vecteur_AC[0]+P_A[0],vecteur_AC[1]+P_A[1]]
            vec_CE=[X[0]-P_C[0],X[1]-P_C[1]]
            theta_prime=[float(np.arctan2(P_C[1],P_C[0])%(2*np.pi)),float((np.arctan2(vec_CE[1],vec_CE[0])-np.arctan2(P_C[1],P_C[0]))%(2*np.pi))]
            a=F(theta[0],theta[1])
            b=F(theta_prime[0],theta_prime[1])
            print(" solution non-uniques:\n la première solution:",theta,"\n","l'autre solution :",theta_prime)
            print("\n vérification de la valeur:\n pour theta,modèle directe nous donne:",a,"\n","pour theta_prime,modèle directe nous donne:",b)
            print("en génerale, on utilise le symétrie pour trouver une autre solution,les deux solition sont symétrique par rappor a la ligne OP(O est le point origine et P est le point de la pose)")
        else : #pas de solution 
            sol = optimize.root(fun, [1, 0], jac=jac, method='hybr')#[1,1]est le point départ pour minimiser la fonction
            print("la pose donnée est hors la porté atteignable")
            print("le point le plus proche de la pose donnée:",sol.x)
            print("la distance entre la pose donné et le point qui est le plus proche de la pose donnée:",np.sqrt(abs(sol.fun)))
            theta="il n'y pas de solution pour ce cas,hors de portée du bras robotique"
            theta_prime="il n'y pas de solution pour ce cas,hors de portée du bras robotique"
    return theta,theta_prime

