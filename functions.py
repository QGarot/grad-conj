import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def display_function(u: np.array, opt: str = "--", debug: bool = False) -> None:
    """
    Affiche le graphe de la fonction dont les valeurs sont stockées dans le vecteur u
    :param debug:
    :param u:
    :param opt:
    :return:
    """
    nb_points = len(u)
    delta = 1 / (nb_points - 1)

    # Abscisse
    # (Le vecteurs u contient nb_points composantes. L'abscisse doit alors être composé de nb_points points (x_i),
    # de façon à afficher les points de coordonnées (x_i, u_i)).
    x = [(i-1) * delta for i in range(1, nb_points+1)]

    # Ordonnée
    y = u

    if debug:
        print("-------  Affichage d'une solution  -------")
        print(">> 'Longueur' de l'abscisse : ", len(x))
        print(">> 'Longueur' de l'ordonnée : ", len(y))
        print("------------------------------------------")

    # Affichage
    plt.plot(x, y, ls=opt, marker=".")
    plt.grid(True)
    plt.title("Graphe de la fonction u")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.show()


def conjugate_gradient_method(A: np.array, b: np.array, eps: float, conditional_loop_param: int, debug: bool = False) -> np.array:
    """
    TODO: faire la version conditionnelle
    Résout le système linéaire Ax = b avec la méthode du gradient conjugué
    Utilité d'une boucle inconditionnelle : étudier la vitesse de convergence de la méthode
    :param A: matrice carrée
    :param b:
    :param conditional_loop_param: entier correspondant au nombre maximal de tour de boucle à effectuer.
    :param debug:
    :param eps: précision
    :return x: solution du système linéaire avec conditions aux bords
    """

    # Initialisation
    n = np.shape(A)[0]
    x = np.zeros(n)
    r = b - np.matmul(A, x)
    p = r

    if debug:
        print("----------- Gradient conjugué  -----------")
        print(">> n = ", n)
        print(">> x = \n", x)
        print(">> r = \n", r)
        print(">> p = \n", p)
        print("")

    if conditional_loop_param != -1:
        kmax = conditional_loop_param
        for k in range(kmax):
            alpha = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(p), A), p)
            x = x + alpha * p
            rk = r
            r = rk - alpha * np.matmul(A, p)

            if np.linalg.norm(r) <= eps:
                break
            beta = np.matmul(np.transpose(r), r) / np.matmul(np.transpose(rk), rk)
            p = r + beta * p
            if debug:
                print("k = ", k)
                print(">>  Distance à la solution = ", np.linalg.norm(np.matmul(A, x) - b))
    else:
        pass

    # Vérification
    if debug:
        print("")
        print("Vérification :")
        print(">> b = ", b)
        print(">> Ax = ", np.matmul(A, x))
        print(">> Distance finale à la solution = ", np.linalg.norm(np.matmul(A, x) - b))
        print("------------------------------------------")

    # Conditions aux bords
    x = np.insert(x, 0, 1)
    x = np.append(x, 0.0)

    return x


def define_linear_system(n: int, k: float, debug: bool = False) -> tuple[np.array, np.array]:
    """
    Retourne le système linéaire associé à la résolution de l'équation différentielle : ...
    :param debug: Vrai si on affiche les matrices, faux sinon
    :param k: paramètre réel du système
    :param n: dimension du système
    :return:
    """
    # Initialisation de la matrice (n*n) A, la matrice colonne (n*1) b et du paramètre h
    A = np.zeros((n, n))
    b = np.zeros(n)
    h = 1 / (n+1)

    # Remplissage des matrices A et b
    for i in range(n):
        if i == 0:
            A[0, 0] = k + 2 / (h ** 2)
            A[0, 1] = - 1 / (h ** 2)
            b[0] = 1 / (h ** 2)  # - 10
        elif i == n - 1:
            A[n - 1, n - 1] = k + 2 / (h ** 2)
            A[n - 1, n - 2] = - 1 / (h ** 2)
            # b[i] = - 10
        else:
            A[i, i - 1] = - 1 / (h ** 2)
            A[i, i] = k + 2 / (h ** 2)
            A[i, i + 1] = - 1 / (h ** 2)
            # b[i] = - 10

    if debug:
        print("------ Définition du système Ax = b ------")
        print(">> A = \n", A)
        print(">> b = \n", b)
        print("------------------------------------------")

    return A, b


def get_lower(A: np.array) -> np.array:
    """
    Retourne la matrice ne contenant que la partie triangulaire inférieure de la matrice A
    :param strict:
    :param A:
    :return:
    """
    n = np.shape(A)[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            L[i, j] = A[i, j]

    return L


def get_upper(A: np.array) -> np.array:
    """
    Retourne la matrice ne contenant que la partie triangulaire supérieure de la matrice A
    :param strict:
    :param A:
    :return:
    """
    n = np.shape(A)[0]
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            U[i, j] = A[i, j]

    return U


def get_diagonal(A: np.array) -> np.array:
    """
    Retourne la matrice ne contenant que la partie diagonale de la matrice A
    :param A:
    :return:
    """
    n = np.shape(A)[0]
    D = np.zeros((n, n))

    for i in range(n):
        D[i, i] = A[i, i]

    return D


def kronecker(i: int, j: int) -> int:
    """
    Symbole delta de Kronecker
    :param i:
    :param j:
    :return:
    """
    if i == j:
        return 1
    else:
        return 0


def inv_lower_triangular(T: np.array) -> np.array:
    """
    TODO: inv_upper_triangular
    Retourne l'inverse d'une matrice triangulaire inversible
    :param T:
    :return:
    """
    n = np.shape(T)[0]
    TM1 = np.zeros((n, n))
    for l in range(n):
        for c in range(l + 1):
            sum = 0
            for k in range(c, l + 1):
                sum = sum + T[l, k] * TM1[k, c]
            TM1[l, c] = (1 / T[l, l]) * (kronecker(l, c) - sum)
    return TM1


def conjugate_gradient_method_ssor(A: np.array, b: np.array, eps: float, conditional_loop_param: int, w: float, debug: bool = False) -> np.array:
    """
    Résout le système linéaire Ax = b avec la méthode du gradient conjugué, en utilisant le préconditionneur SSOR
    :param A:
    :param b:
    :param eps:
    :param conditional_loop_param:
    :param w:
    :param debug:
    :return:
    """

    # Initialisation
    D = get_diagonal(A)
    E = - get_lower(A)
    T = (1 / sqrt(w * (2 - w))) * (D - w * E) * np.sqrt(D)
    T_inv = inv_lower_triangular(T)
    T_transpose_inv = np.linalg.inv(np.transpose(T))
    prec = np.matmul(T_transpose_inv, T_inv)

    # Résolution du système : P^-1 * A *x = P^-1 * b
    z = conjugate_gradient_method(np.matmul(prec, A), np.matmul(prec, b), eps, conditional_loop_param, debug)
    return z
