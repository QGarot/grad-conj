import functions


if __name__ == "__main__":
    A, b = functions.define_linear_system(50, 10, True)
    x = functions.conjugate_gradient_method(A, b, 1e-12, 100, True)
    functions.display_solution(x, "--", True)
