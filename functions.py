import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from typing import Union


def display_function(u: np.array, debug: bool = False) -> None:
    """
    Affiche le graphe de la fonction dont les valeurs "discrètes" sont stockées dans le vecteur u
    :param u:
    :param debug:
    :return:
    """
    nb_points = len(u)
    delta = 1 / (nb_points - 1)

    # Abscisse
    # (Le vecteurs u contient nb_points composantes. L'abscisse doit alors être composé de nb_points points (x_i),
    # de façon à afficher les points de coordonnées (x_i, u_i)).
    x = [(i - 1) * delta for i in range(1, nb_points + 1)]

    # Ordonnée
    y = u

    if debug:
        print("-------  Affichage d'une solution  -------")
        print(">> 'Longueur' de l'abscisse :", len(x))
        print(">> 'Longueur' de l'ordonnée :", len(y))
        print("------------------------------------------")

    # Affichage
    plt.plot(x, y, ls="-", marker=".")
    plt.grid(True)
    plt.title("Graphe de la fonction $u$")
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.show()


def conjugate_gradient_method(A: np.array, b: np.array, eps: float, kmax: int, debug: bool = False, convergence: bool = False) -> Union[np.array, list[np.array, list[list, list]]]:
    """
    Retourne la solution du système linéaire Ax = b en utilisant la méthode du gradient conjugué.
    Utilité de l'usage d'une boucle inconditionnelle : étudier la vitesse de convergence de la méthode.
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param eps: précision
    :param kmax: entier correspondant au nombre maximal de tour de boucle à effectuer
    :param debug:
    :param convergence:
    Si ce paramètre vaut True, la fonction retourne la solution ET les informations permettant
    d'étudier la convergence de la méthode, à savoir :
        - une liste contenant les indices de chaque étape (tours de boucle
        - une liste contenant || Ax_k - b ||, où x_k est l'approximation de la solution du système linéaire à la k-ème
        étape. Plus cette valeur est proche de 0, plus x_k est proche de la solution exacte.
    :return x: solution du système linéaire avec conditions aux bords
    """
    n = np.shape(A)[0]

    # Initialisation
    x = np.zeros(n)
    r = b - np.matmul(A, x)
    p = r

    # Initialisation des paramètres pour étudier la convergence
    # liste des indices
    indices = []
    # liste des marges d'erreur, c'est à dire des || Ax_k - b ||
    margins_of_error = []

    if debug:
        print("----------- Gradient conjugué  -----------")
        print(">> n =", n)
        print(">> x =\n", x)
        print(">> r =\n", r)
        print(">> p =\n", p)
        print("")

    for k in range(kmax):
        alpha = np.matmul(np.transpose(r), r) / np.matmul(np.matmul(np.transpose(p), A), p)
        x = x + alpha * p
        rk = r
        r = rk - alpha * np.matmul(A, p)
        res = np.linalg.norm(np.matmul(A, x) - b)

        if convergence:
            indices.append(k)
            margins_of_error.append(res)

        if np.linalg.norm(r) <= eps:
            break

        beta = np.matmul(np.transpose(r), r) / np.matmul(np.transpose(rk), rk)
        p = r + beta * p

        if debug:
            print("k =", k)
            print(">>  Distance à la solution =", np.linalg.norm(np.matmul(A, x) - b))

    # Vérification
    if debug:
        print("")
        print("Vérification :")
        print(">> b =", b)
        print(">> Ax =", np.matmul(A, x))
        print(">> Distance finale à la solution =", np.linalg.norm(np.matmul(A, x) - b))
        print("------------------------------------------")

    # Conditions aux bords
    x = np.insert(x, 0, 1)
    x = np.append(x, 0.0)

    # Si on veut étudier la convergence, on retourne les données nécessaires pour afficher le graphe de || Ax_k - b ||
    # en fonction de k
    if convergence:
        return [x, [indices, margins_of_error]]
    else:
        return x


def define_linear_system(n: int, k: float, debug: bool = False) -> tuple[np.array, np.array]:
    """
    Retourne le système linéaire associé à la résolution de l'équation différentielle : ...
    :param n: dimension du système
    :param k: paramètre réel du système
    :param debug:
    :return:
    """
    # Initialisation de la matrice (n*n) A, la matrice colonne (n*1) b et du paramètre h
    A = np.zeros((n, n))
    b = np.zeros(n)
    h = 1 / (n + 1)

    # Remplissage des matrices A et b
    for i in range(n):
        if i == 0:
            A[0, 0] = k + 2 / (h ** 2)
            A[0, 1] = - 1 / (h ** 2)
            b[0] = 1 / (h ** 2)
        elif i == n - 1:
            A[n - 1, n - 1] = k + 2 / (h ** 2)
            A[n - 1, n - 2] = - 1 / (h ** 2)
        else:
            A[i, i - 1] = - 1 / (h ** 2)
            A[i, i] = k + 2 / (h ** 2)
            A[i, i + 1] = - 1 / (h ** 2)

    if debug:
        print("------ Définition du système Ax = b ------")
        print(">> A =\n", A)
        print(">> b =\n", b)
        print("------------------------------------------")

    return A, b


def get_lower_strict(A: np.array) -> np.array:
    """
    Retourne la matrice ne contenant que la partie triangulaire inférieure STRICTE de la matrice A
    :param A:
    :return:
    """
    n = np.shape(A)[0]
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i):
            L[i, j] = A[i, j]

    return L


def get_upper_strict(A: np.array) -> np.array:
    """
    Retourne la matrice ne contenant que la partie triangulaire supérieure STRICTE de la matrice A
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


def solve_lower(L: np.array, b: np.array) -> np.array:
    """
    Retourne la solution du système linéaire Lx = b
    :param L: matrice triangulaire inférieure inversible
    :param b: second membre
    :return:
    """
    n = np.shape(L)[0]
    x = np.zeros(n)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] = x[i] - L[i, j] * x[j]
        x[i] = x[i] / L[i, i]
    
    return x


def solve_upper(U: np.array, b: np.array) -> np.array:
    """
    Retourne la solution du système linéaire Ux = b
    :param U: matrice triangulaire supérieure inversible
    :param b: second membre
    :return:
    """
    n = np.shape(U)[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] = x[i] - U[i, j] * x[j]
        x[i] = x[i] / U[i, i]
    
    return x


def conjugate_gradient_method_ssor_v2(A: np.array, b: np.array, eps: float, kmax: int, w: float, debug: bool = False, convergence: bool = False) -> Union[np.array, list[np.array, list]]:
    """
    Retourne la solution du système linéaire Ax = b en utilisant la méthode du gradient conjugué avec préconditionnement
    SSOR
    Utilité de l'usage d'une boucle inconditionnelle : étudier la vitesse de convergence de la méthode.
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param eps: précision
    :param kmax: entier correspondant au nombre maximal de tour de boucle à effectuer
    :param w: paramètre du préconditionneur SSOR
    :param debug:
    :param convergence:
    Si ce paramètre vaut True, la fonction retourne la solution ET les informations permettant
    d'étudier la convergence de la méthode, à savoir :
        - une liste contenant les indices de chaque étape (tours de boucle
        - une liste contenant || Ax_k - b ||, où x_k est l'approximation de la solution du système linéaire à la k-ème
        étape. Plus cette valeur est proche de 0, plus x_k est proche de la solution exacte.
    :return x: solution du système linéaire avec conditions aux bords
    """
    n = np.shape(A)[0]

    # Initialisation
    x = np.zeros(n)
    r = b - np.matmul(A, x)

    # (Modifications pour SSOR)
    D = get_diagonal(A)
    DM12 = np.zeros((n, n))

    for i in range(n):
        DM12[i, i] = 1 / sqrt(D[i, i])

    E = - get_lower_strict(A)
    T = (1 / sqrt(w * (2 - w))) * np.matmul((D - w * E), DM12)
    T_transpose = np.transpose(T)

    y = solve_lower(T, r)
    p = solve_upper(T_transpose, y)

    z = p

    # Initialisation des paramètres pour étudier la convergence
    # liste des indices
    indices = []
    # liste des marges d'erreur, c'est à dire des || Ax_k - b ||
    margins_of_error = []

    if debug:
        print("----------- Gradient conjugué  -----------")
        print(">> n =", n)
        print(">> x =\n", x)
        print(">> r =\n", r)
        print(">> p =\n", p)
        print("")

    for k in range(kmax):
        alpha = np.matmul(np.transpose(r), z) / np.matmul(np.matmul(np.transpose(p), A), p)
        x = x + alpha * p
        rk = r
        r = rk - alpha * np.matmul(A, p)
        res = np.linalg.norm(np.matmul(A, x) - b)

        if convergence:
            indices.append(k)
            margins_of_error.append(res)

        if np.linalg.norm(r) <= eps:
            break
        
        # (Modification pour SSOR)
        zk = z
        y = solve_lower(T, r)
        z = solve_upper(T_transpose, y)

        beta = np.matmul(np.transpose(r), z) / np.matmul(np.transpose(rk), zk)
        p = z + beta * p

        if debug:
            print("k =", k)
            print(">>  Distance à la solution =", np.linalg.norm(np.matmul(A, x) - b))

    # Vérification
    if debug:
        print("")
        print("Vérification :")
        print(">> b =", b)
        print(">> Ax =", np.matmul(A, x))
        print(">> Distance finale à la solution =", np.linalg.norm(np.matmul(A, x) - b))
        print("------------------------------------------")

    # Conditions aux bords
    x = np.insert(x, 0, 1)
    x = np.append(x, 0.0)

    # Si on veut étudier la convergence, on retourne les données nécessaires pour afficher le graphe de || Ax_k - b ||
    # en fonction de k
    if convergence:
        return [x, [indices, margins_of_error]]
    else:
        return x


def conjugate_gradient_method_ssor(A: np.array, b: np.array, eps: float, kmax: int, w: float, debug: bool = False, convergence: bool = False) -> Union[np.array, list[np.array, list]]:
    """
    TODO: pas besoin de passer par le calcul direct de l'inverse du préconditionneur.
    Retourne la solution du système linéaire Ax = b en utilisant la méthode du gradient conjugué avec préconditionnement
    SSOR.
    :param A: matrice carrée symétrique définie positive
    :param b: second membre
    :param eps: précision
    :param kmax: entier correspondant au nombre maximal de tours de boucle
    :param w: paramètrage SSOR
    :param debug:
    :param convergence:
    Si ce paramètre vaut True, la fonction retourne la solution ET les informations permettant
    d'étudier la convergence de la méthode, à savoir :
        - une liste contenant les indices de chaque étape (tours de boucle
        - une liste contenant || Ax_k - b ||, où x_k est l'approximation de la solution du système linéaire à la k-ème
        étape. Plus cette valeur est proche de 0, plus x_k est proche de la solution exacte.
    :return x: solution du système linéaire avec conditions aux bords
    :return:
    """
    # Initialisation des variables nécessaires pour calculer l'inverse du préconditionneur
    n = np.shape(A)[0]
    D = get_diagonal(A)
    DM12 = np.zeros((n, n))

    for i in range(n):
        DM12[i, i] = 1 / sqrt(D[i, i])

    E = - get_lower_strict(A)
    T = (1 / sqrt(w * (2 - w))) * (D - w * E) * DM12
    T_inv = inv_lower_triangular(T)
    T_transpose_inv = np.linalg.inv(np.transpose(T))

    # Calcul de l'inverse du préconditionneur
    prec = np.matmul(T_transpose_inv, T_inv)

    # Résolution du système : P^-1 * A *x = P^-1 * b
    return conjugate_gradient_method(np.matmul(prec, A), np.matmul(prec, b), eps, kmax, debug, convergence)
