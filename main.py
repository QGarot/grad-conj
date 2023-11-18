import analysis


if __name__ == "__main__":
    # A, b = functions.define_linear_system(50, 10, True)
    # x = functions.conjugate_gradient_method(A, b, 1e-12, 100, True)
    # functions.display_function(x, "--", True)
    analysis.multi_display_k([-1, 0, 5, 10, 15, 100], 50)

