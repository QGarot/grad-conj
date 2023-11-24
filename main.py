import analysis
import functions
import numpy as np


if __name__ == "__main__":
    n = 10
    k = 50
    A, b = functions.define_linear_system(n, k, True)
    # ----------------------------------------------------------------------------------------
    # 1. Gradient conjugué :
    w = 1.9
    epsilon = 1e-9
    iter_max = 60
    # 1.a) avec ssor
    x1 = functions.conjugate_gradient_method_ssor(A, b, epsilon, iter_max, w, True, False)
    functions.display_function(x1, True)
    # 1.b) sans SSOR
    x2 = functions.conjugate_gradient_method(A, b, epsilon, iter_max, True, False)
    functions.display_function(x2, True)
    # ----------------------------------------------------------------------------------------
    # 2. Comparaison des solutions en fonction de la taille du système linéaire
    linear_system_sizes = [5, 10, 50]
    k = 300
    analysis.multi_display_dimensions(linear_system_sizes, k)
    # ----------------------------------------------------------------------------------------
    # 3. Comparaison des solutions en fonctions du paramètre k intervenant dans le système
    # 3.a) k > 0
    n = 50
    params1 = [0, 40, 80, 120, 160, 200]
    analysis.multi_display_k(params1, n)
    # 3.b) k << 0
    params2 = [-100, -200, -300, -400, -500]
    analysis.multi_display_k(params2, n)
    # ----------------------------------------------------------------------------------------
    # 4. Comparaison entre la convergence de la méthode du gradient conjugué sans
    #    préconditionnement avec celle de la méthode du gradient conjugué avec précond. SSOR.
    epsilon = 1e-9
    iter_max = 60
    with_log = True
    analysis.compare_convergence(A, b, epsilon, iter_max, with_log, True)
    # ----------------------------------------------------------------------------------------
    # 5. Comparaison de la convergence de la méthode du gradient conjugué en fonction du
    #    paramètre du préconditionneur SSOS (w)
    params = [0.1, 0.6, 0.9, 1.2, 1.8, 1.99]
    analysis.compare_convergence_ssor_w(A, b, params)

