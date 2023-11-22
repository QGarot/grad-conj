import analysis
import functions


if __name__ == "__main__":
    A, b = functions.define_linear_system(20, 50, True)
    # Gradient conjugué SANS SSOR
    #x = functions.conjugate_gradient_method_ssor(A, b, 1e-8, 100, True, False)

    # Gradient conjugué AVEC SSOR
    #x = functions.conjugate_gradient_method_ssor(A, b, 1e-8, 100, 1.89, True, False)
    #functions.display_function(x, "-", True)

    # analysis.multi_display_dimensions([4, 50], 100)
    # analysis.compare_convergence(A, b, 1e-10, 500, True, True)
    analysis.compare_convergence_ssor(A, b, [0.05, 0.5, 1.2, 1.99])
