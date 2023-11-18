import numpy as np
import matplotlib.pyplot as plt


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
    TODO: faire la version inconditionnelle
    Résout le système linéaire Ax = b avec la méthode du gradient conjugué
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
    h = 1/(n+1)

    # Remplissage des matrices A et b
    for i in range(n):
        if i == 0:
            A[0, 0] = k + 2/(h**2)
            A[0, 1] = - 1/(h**2)
            b[0] = 1/(h**2)
        elif i == n - 1:
            A[n - 1, n - 1] = k + 2/(h ** 2)
            A[n - 1, n - 2] = - 1/(h ** 2)
        else:
            A[i, i - 1] = - 1/(h ** 2)
            A[i, i] = k + 2/(h ** 2)
            A[i, i + 1] = - 1/(h ** 2)

    if debug:
        print("------ Définition du système Ax = b ------")
        print(">> A = \n", A)
        print(">> b = \n", b)
        print("------------------------------------------")

    return A, b
