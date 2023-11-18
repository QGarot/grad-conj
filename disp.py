import numpy as np
import matplotlib.pyplot as plt


def display(u: np.array, opt: str = "--"):
    """
    u_i = u(x_i), où u est la fonction solution.
    Cette fonction affiche le graphe de la fonction u.

    Note : u -> taille de u -> taille du système linéaire
           -> nb de points de discrétisations -> on connait les paramètres pour l'abscisses
    """

    n = len(u)
    h = 1 / (n - 1)

    # abscisse
    x = [(i-1) * h for i in range(1, n+1)]
    print(len(x))

    # ordonnée
    y = u
    print(len(y))

    # affichage
    plt.plot(x, y, ls=opt, marker=".")
    plt.grid(True)
    plt.title("Graphe de la fonction u")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()

# u = np.array([0,2,4,6,8,10,10,10,10,10,8,10,10,2,5,2,3,55,6,3,2,4,5,5,2,0])
# disp(u)

