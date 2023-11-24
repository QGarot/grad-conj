import functions
import matplotlib.pyplot as plt
import numpy as np


def multi_display_dimensions(dimensions: list[int], k: int) -> None:
    """
    Affiche sur un même graphique les solutions approchées du système linéaire en fonction de sa dimension
    Exemple :
        >> multi_display_dimensions([5, 10, 50], 5)
        Cette instruction affiche les solutions des systèmes linéaires de dimensions 5, 10 et 50, tous paramétrés avec
        k = 5
    :param dimensions: liste des dimensions
    :param k: k fixé
    :return:
    """
    for dim in dimensions:
        A, b = functions.define_linear_system(dim, k)
        x = functions.conjugate_gradient_method(A, b, 1e-9, 50)

        nb_points = len(x)
        delta = 1 / (nb_points - 1)
        plt.plot([(i-1) * delta for i in range(1, nb_points+1)], x, ls="-", marker=".", label="n = " + str(dim))

    plt.legend()
    # Titre affiché dans le rapport
    #plt.title("Solutions du système linéaire $Ax - b$ en fonction de sa dimension ($k$ fixé à " + str(k) + ")")
    plt.grid(True)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.show()


def multi_display_k(params: list[int], n: int) -> None:
    """
    Affiche sur un même graphique les solutions approchées du système linéaire en fonction du paramètre k
    Exemple :
        >> multi_display_k([5, 10, 50], 50)
        Cette instruction affiche les solutions des systèmes linéaires de dimension 50 paramétrés avec k = 5, 10, 50.
    :param params: liste des paramètres k
    :param n: dimension du système linéaire fixée
    :return:
    """
    for k in params:
        A, b = functions.define_linear_system(n, k)
        x = functions.conjugate_gradient_method(A, b, 1e-9, 100)

        nb_points = len(x)
        delta = 1 / (nb_points - 1)
        plt.plot([(i-1) * delta for i in range(1, nb_points+1)], x, ls="-", marker=".", label="k = " + str(k))

    plt.legend()
    # Titre affiché dans le rapport
    #plt.title("Solutions du système linéaire $Ax = b$ en fonction du paramètre $k$ ($n = " + str(n) + "$)")
    plt.grid(True)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.show()


def compare_convergence_ssor_w(A: np.array, b: np.array, params: list[float]) -> None:
    """
    Affiche sur un même graphique les convergences de la méthode du gradient conjugué préconditionné (SSOR) en fonction
    de son paramètre w. Tests effectués avec une bonne précision.
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param params: liste contenant les paramètres w à tester
    :return:
    """
    for w in params:
        indices, margins_of_error = functions.conjugate_gradient_method_ssor(A, b, 1e-5, 100, w, False, True)[1]
        plt.plot(indices, margins_of_error, ls="-", marker=".", label="w = " + str(w))

    plt.grid(True)
    plt.legend()
    # Titre affiché dans le rapport
    #plt.title("Comparaison des convergences de la méthode du\n gradient conjugué préconditionné (SSOR) en fonction de $w$\n")
    plt.xlabel("$k$")
    plt.ylabel("$\Vert Ax_k - b \Vert$")
    plt.show()


def compare_convergence(A: np.array, b: np.array, eps: float, kmax: int, log: bool, debug: bool = False) -> None:
    """
    Affiche sur un même graphique les convergences de la méthode du gradient conjugué SANS et AVEC préconditionnement
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param eps: précision
    :param kmax: nombre de tours de boucle maximal à effectuer lors des deux appels
    :param log: vaut True si on affiche le log10 des marges d'erreur
    :param debug:
    :return:
    """
    indices, margins_of_error = functions.conjugate_gradient_method(A, b, eps, kmax, debug, True)[1]
    indices_ssor, margins_of_error_ssor = functions.conjugate_gradient_method_ssor(A, b, eps, kmax, 1, debug, True)[1]

    if debug:
        print("")
        print(">> indices =", indices)
        print(">> margins_of_error =", margins_of_error)
        print("")
        print(">> indices_ssor =", indices_ssor)
        print(">> margins_of_error_ssor =", margins_of_error_ssor)

    if log:
        plt.plot(indices, np.log10(margins_of_error), ls="-", marker=".", label="Sans SSOR")
        plt.plot(indices_ssor, np.log10(margins_of_error_ssor), ls="-", marker=".", label="Avec SSOR")
    else:
        plt.plot(indices, margins_of_error, ls="-", marker=".", label="Sans SSOR")
        plt.plot(indices_ssor, margins_of_error_ssor, ls="-", marker=".", label="Avec SSOR")

    plt.grid(True)
    plt.legend()
    # Titre affiché sur le rapport...
    #plt.title("Comparaison de la convergence de la méthode du gradient conjugué sans préconditionnement avec celle
    # de la méthode du gradient avec préconditionnement \n")
    plt.xlabel("$k$")
    if log:
        plt.ylabel("$log_{10}(\Vert Ax_k - b \Vert)$")
    else:
        plt.ylabel("$\Vert Ax_k - b \Vert$")

    plt.show()
