import analysis
import functions
import numpy as np


if __name__ == "__main__":
    A, b = functions.define_linear_system(50, 50, True)
    # Gradient conjugué SANS SSOR
    #x = functions.conjugate_gradient_method_ssor(A, b, 1e-8, 100, True, False)

    # Gradient conjugué AVEC SSOR
    """x = functions.conjugate_gradient_method_ssor_v2(A, b, 1e-8, 100, 1.89, True, False)
    functions.display_function(x, True)
    
    analysis.compare_convergence(A, b, 1e-10, 500, False, True)"""
    
    analysis.compare_convergence_ssor_w(A, b, [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 1.99])
    analysis.compare_convergence(A, b, 1e-10, 500, True, True)

    """analysis.multi_display_dimensions([5, 10, 50], 150)
    analysis.multi_display_k([1, 10, 50, 100, 150], 60)
    analysis.compare_convergence(A, b, 1e-10, 500, True, True)
    analysis.compare_convergence_ssor(A, b, [1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 1.99])"""
