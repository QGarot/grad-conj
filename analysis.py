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
        x = functions.conjugate_gradient_method(A, b, 0.0001, 50)

        nb_points = len(x)
        delta = 1 / (nb_points - 1)
        plt.plot([(i-1) * delta for i in range(1, nb_points+1)], x, ls="-", marker=".", label="n = " + str(dim))

    plt.legend()
    plt.title("Solutions du système linéaire Ax = b en fonction de sa dimension")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u(x)")
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
        x = functions.conjugate_gradient_method(A, b, 0.001, 50)

        nb_points = len(x)
        delta = 1 / (nb_points - 1)
        plt.plot([(i-1) * delta for i in range(1, nb_points+1)], x, ls="-", marker=".", label="k = " + str(k))

    plt.legend()
    plt.title("Solutions du système linéaire Ax = b en fonction du paramètre k")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()


def compare_convergence(A: np.array, b: np.array, eps: float, kmax: int, debug: bool = False) -> None:
    """
    Affiche sur un même graphique les convergences de la méthode du gradient conjugué SANS et AVEC préconditionnement
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param eps: précision
    :param kmax: nombre de tours de boucle maximal à effectuer lors des deux appels
    :param debug:
    :return:
    """
    indices, margins_of_error = functions.conjugate_gradient_method(A, b, eps, kmax, debug, True)[1]
    indices_ssor, margins_of_error_ssor = functions.conjugate_gradient_method_ssor(A, b, eps, kmax, 1, debug, True)[1]

    if debug:
        print("")
        print(">> indices = ", indices)
        print(">> margins_of_error = ", margins_of_error)
        print("")
        print(">> indices_ssor = ", indices_ssor)
        print(">> margins_of_error_ssor = ", margins_of_error_ssor)

    plt.plot(indices, margins_of_error, ls="-", marker=".", label="Sans SSOR")
    plt.plot(indices_ssor, margins_of_error_ssor, ls="-", marker=".", label="Avec SSOR")
    plt.grid(True)
    plt.legend()
    plt.title("Comparaison convergence")
    plt.xlabel("k")
    plt.ylabel("|| Ax_k - b ||")
    plt.show()
