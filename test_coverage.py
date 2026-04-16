from numba import njit


# Check if Numba is installed and works
@njit
def f():
    return 1


print(f())
