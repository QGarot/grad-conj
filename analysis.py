import functions
import matplotlib.pyplot as plt


def multi_display_dimensions(dimensions: list[int], k: int) -> None:
    """
    Affiche sur un même graphique les solutions approchées du système linéaire en fonction de sa dimension
    :param dimensions:
    :param k: k fixé
    :return:
    """
    for dim in dimensions:
        A, b = functions.define_linear_system(dim, k)
        x = functions.conjugate_gradient_method(A, b, 0.0001, 50)

        nb_points = len(x)
        delta = 1 / (nb_points - 1)
        plt.plot([(i-1) * delta for i in range(1, nb_points+1)], x, ls="--", marker=".")

    plt.grid(True)
    plt.show()


def multi_display_k(params: list[int], n: int) -> None:
    """
    Affiche sur un même graphique les solutions approchées du système linéaire en fonction du paramètre k
    :param params:
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


def compare_convergence() -> None:
    pass
