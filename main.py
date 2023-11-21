import analysis
import functions


if __name__ == "__main__":
    A, b = functions.define_linear_system(50, 5, True)
    #x = functions.conjugate_gradient_method(A, b, 1e-12, 100, True)
    #functions.display_function(x, "-", True)
    # analysis.multi_display_k([-9, -10, -9.1, -9.3, -9.6, -11], 50)
    #D = functions.get_diagonal(A)
    #U = functions.get_upper(A)
    #L = functions.get_lower(A)
    #DM1 = np.zeros((np.shape(A)[0], np.shape(A)[0]))
    #for i in range(np.shape(A)[0]):
    #    DM1[i, i] = 1 / D[i, i]
    x = functions.conjugate_gradient_method_ssor(A, b, 1e-12, 1000, 0.5, True)
    functions.display_function(x, "-", True)
