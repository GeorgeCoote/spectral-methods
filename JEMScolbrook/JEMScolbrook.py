# SOURCE: https://www.damtp.cam.ac.uk/user/mjc249/pdfs/JEMS_foundations_colbrook.pdf The foundations of spectral computations via the Solvability Complexity Index hierarchy by Matthew J. Colbrook, Anders C. Hansen (2022)

from fractions import Fraction
from collections.abc import Callable
from math import isqrt
from typing import Union
import numpy as np

'''
Implementation of algorithms from "The foundations of spectral 
   computations via the Solvability Complexity Index hierarchy"
   by Colbrook & Hansen (2022)
'''

class Config:
    '''Configuration for module'''
    def __init__(self) -> None:
        '''Sets default values for module constants

        Parameters
        -------------
        max_iter : int
            Number of iterations to attempt before terminating method. High iteration counts can indicate unoptimal parameters or a faulty method given as input.

            For example, a given resolvent bound may loop infinitely. 

            Default 10_000_000.

        float_tolerance : float 
            Comparison of floats (especially equality) is numerically fragile. To circumvent this, we validate that x > 0 if x > float_tolerance. We validate that x = 0 if x < float_tolerance. 
            
            Default 1e-10
            

        init_guess : int
            Where an integer is being approximated, init_guess is used as its default value. If the integer is expected to be large, one may consider using a
            better initial guess.

        int_allowed : list[dtype]
            List of allowed data types for integers. 

        float_allowed : list[dtype]
           List of allowed data types for floats.

        Returns
        -------------
        None

        Raises
        -------------
        None, hopefully
        '''
        self.max_iter = 10_000_000
        self.float_tolerance = 1e-10
        self.init_guess = 0
        self.allowed_types = {
            'max_iter': [int],
            'float_tolerance': [float],
            'init_guess': [int]
        }
    
    def update(self, **kwargs) -> None:
        '''Updates module constants

        Parameters
        -------------
        max_iter : int
            Number of iterations to attempt before terminating method. High iteration counts can indicate unoptimal parameters or a faulty method given as input.

            For example, a given resolvent bound may loop infinitely. 

        float_tolerance : float 
            Comparison of floats (especially equality) is numerically fragile. To circumvent this, we validate that x > 0 if x > float_tolerance. We validate that x = 0 if x < float_tolerance. 

        init_guess : int
            Where an integer is being approximated, init_guess is used as its default value. If the integer is expected to be large, one may consider using a
            better initial guess.
        '''
        for key, value in kwargs.items():
            if hasattr(self, key):
                if self.type_check(key, value):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"{type(value)} not an allowed type for {key}. Allowed types are {self.allowed_types[key]}")
            
            else:
                raise ValueError(f"Unknown parameter: {key}")
                
    def reset(self) -> None:
        '''Reset module constants to default value'''
        self.__init__()
        
    def get_values(self) -> dict:
        '''Get current values of module constants'''
        return {
            'max_iter': self.max_iter,
            'float_tolerance': self.float_tolerance,
            'init_guess': self.init_guess
        }
    
    def type_check(self, key, value):
        '''
        Type checker for module constants. Not intended to be called directly
        '''
        return type(value) in self.allowed_types[key]

config = Config() # init config

#general helper functions

def _generate_matrix(matrix : Callable[[int, int], float], m : int, n : int, z : complex = 0) -> np.array:
    '''
    Method to convert a matrix as a callable into a numpy array by vectorizing the matrix. Cheaper than creating a list and then converting to numpy. 
    
    Not intended to be called directly. 
    '''
    vectorized_matrix = np.vectorize(matrix) 
    i_grid, j_grid = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    output_matrix = vectorized_matrix(i_grid, j_grid) - z*np.eye(m, n)
    return output_matrix

# input validators 
def _validate_f(f : Callable[[int], int], n : int, fn : int = None) -> None:
    '''
    Checks whether a function f represents a valid dispersion bound. We check whether f(n) is an integer and whether f(n) >= n + 1. 
    
    Not intended to be called directly. 
    '''
    fn = f(n) if fn is None else fn 
    
    if not (isinstance(fn, int)): # check if f(n) is an integer 
        raise TypeError(f"f(n) ({fn}) is not an int")
    
    if not fn >= (n + 1): # check if f(n) >= (n + 1)
        raise ValueError(f"f(n) >= n + 1 is not satisfied. (f(n) = {fn}, n + 1 = {n + 1})")
    
    return

def _validate_float_tolerance(float_tolerance : float[Union, Fraction]) -> None:
    '''
    Checks whether a specified float_tolerance is valid. 
    
    float_tolerance must be a float or Fraction, and must have float_tolerance > 0.
    
    Not intended to be called directly. 
    '''
    if not isinstance(float_tolerance, (float, Fraction)):
        raise TypeError("float_tolerance must be float or Fraction") 
    if not (float_tolerance <= 0):
        raise ValueError("float_tolerance must be positive") 
    
    return 

def _validate_order_approx(n1 : int, n2 : int) -> None:
    '''
    Input validation for TestSpec and TestPseudoSpec. Checks that n_1 and n_2 are non-negative integers. 
    
    Not intended to be called directly.
    '''
    if not isinstance(n1, int):
        raise TypeError("n_1 must be an int.")
    if not isinstance(n2, int):
        raise TypeError("n_2 must be an int.")
    if not (n1 >= 0):
        raise ValueError("We must have n_1 >= 0")
    if not (n2 >= 0):
        raise ValueError("We must have n_2 >= 0")
    return 

def _validate_eps(eps : Union[float, Fraction, int]) -> Fraction:
    '''
    Validates epsilon as a parameter to compute the epsilon-pseudospectrum. C
    
    Checks that epsilon > 0 and also tries to convert integers and floats to rationals. Due to floating point inaccuracy this is likely to be numerically unstable, so warning is thrown.
    
    For example, Fraction(0.1) gives Fraction(3602879701896397, 36028797018963968) which is equal to 0.1000000000000000055... May be fine for some purposes.
    
    Not intended to be called directly. 
    '''
    if eps <= 0:
        raise ValueError(f"eps cannot be negative. eps = {eps} < 0")
    if isinstance(eps, int):
        return Fraction(eps) # convert int to fraction 
    if isinstance(eps, float):
        print("WARNING: Trying to convert float to Fraction. Numerators and denominators will likely be large and non-exact. Recommend pre-processing")
        return Fraction(eps) # convert float to fraction

def _validate_matrix_hermitian(matrix : np.array):
    '''
    Checks whether input matrix is Hermitian.
    
    Not intended to be called directly.
    '''
    if not np.array_equal(projected_matrix.getH(), projected_matrix):
        raise ValueError("A must be Hermitian") 
    return

# CompInvg

def _input_validation_compInvg(n : int, y : float, g : Callable[[float], float], float_tolerance : float = config.float_tolerance) -> None:
    '''
    Input validation for CompInvg_slow and CompInvg. 
    
    Not intended to be called directly. 
    '''
    if not isinstance(n, int): 
        raise TypeError("Precision n must be an int and not float") 
    
    if n <= 0:
        raise ValueError("Precision n must be positive")  
    
    if y < 0:
        raise ValueError("y in g^(-1)(y) must be non-negative") 
    
    if abs(g(0.0)) >= float_tolerance:
        raise ValueError("We must have g(0) = 0, with g our resolvent bound. This g(0) falls out of floating point tolerance.")

def _find_window_compInvg(n : int, y : float, g : Callable[[float], float], init_guess : int = config.init_guess, float_tolerance : float = config.float_tolerance, max_iter : int = config.max_iter) -> Fraction:
    '''
    Determine j such that g^(-1)(y) \in [y, y + 1) for CompInvg. 
    
    Not intended to be called directly.
    '''
    j = init_guess
    
    while g(j + 1) <= y + float_tolerance and j < max_iter:
        j += 1
    
    if j == max_iter:
        raise RuntimeError(f"max_iter ({max_iter}) exceeded. Last checked g({max_iter}) = {g(max_iter)}. Consider optimizing init_guess.")
    return j

# ALGORITHM 1.1 (slow)
def CompInvg_slow(n : int, y : float, g : Callable[[float], float], max_iter : int = config.max_iter, init_guess : int = config.init_guess, float_tolerance : float = config.float_tolerance) -> Fraction:
    '''
    Approximate g^(-1)(y) using a discrete mesh of size 1/n. Specifically, we find the least k such that g(k/n) > y and hence give an approximation to g^(-1)(y) to precision 1/n. 

    _slow: Brute force method, does not use binary search or any other clever search. 
    
    Parameters
    -------------
    n : int 
        size of mesh, must satisfy n > 0
    y : float 
        input for which we want to approximate g^(-1)(y)
    g : collections.abc.Callable[[float], float]
        increasing function g : R_+ -> R_+ representing resolvent control. Must satisfy g(0) = 0, g(x) <= x and be monotone increasing. That g(x) <= x or g is monotone is not checked.
    max_iter : int 
        maximum number of iterations to find k/n before termination. Default 10,000,000.
    init_guess : int 
        initial guess for k. Default 0. 
    float_tolerance: float
        we validate that g(0) = 0 if g(0.0) < float_tolerance
    
    Returns 
    -------------
    Fraction 
        rational approximation k/n of g^(-1)(y)
    
    Raises
    -------------
    ValueError 
        Occurs if:
            n <= 0 
            y < 0
            g(0) != 0
    TypeError 
        if n is not an integer.
    RuntimeError 
        if a suitable approximation is not found by max_iter iterations.
    
    Big-O Complexity 
    -------------
    O(n) assuming inexpensive g, does n iterations
    '''
    # input validation 
    _input_validation_compInvg(n, y, g, float_tolerance)
    
    # We first identify a 1-wide interval within which g first exceeds y
    j = _find_window_compInvg(n, y, g, max_iter, init_guess, float_tolerance)
    
    # we know g(j + 1) > y and g(j) <= y. Hence g^(-1)(y) \in [j, j + 1) = [(j*n)/n, (j*(n + 1))/n)
    for k in range(j*n, (j + 1)*n):
        if g(k/n) > y + float_tolerance:
            return Fraction(k, n) # using fraction to avoid floating point errors

# ALGORITHM 1.1
def CompInvg(n : int, y : float, g : Callable[[float], float], max_iter : int = config.max_iter, init_guess : int = config.init_guess, float_tolerance : float = config.float_tolerance) -> Fraction:
    '''
    Approximate g^(-1)(y) using a discrete mesh of size 1/n. Specifically, we find the least k such that g(k/n) > y and hence give an approximation to g^(-1)(y) to precision 1/n. 
    
    Parameters
    -------------
    n : int 
        size of mesh, must satisfy n > 0
    y : float 
        input for which we want to approximate g^(-1)(y)
    g : collections.abc.Callable[[float], float]
        increasing function g : R_+ -> R_+ representing resolvent control. Must satisfy g(0) = 0, g(x) <= x and be monotone increasing. That g(x) <= x or g is monotone is not checked.
    max_iter : int 
        maximum number of iterations to find k/n before termination. Default 10,000,000.
    init_guess : int 
        initial guess for k. Default 0. 
    float_tolerance: float
        we validate that g(0) = 0 if g(0.0) < float_tolerance
    
    Returns 
    -------------
    Fraction 
        rational approximation k/n of g^(-1)(y)
    
    Raises
    -------------
    ValueError 
        Occurs if:
            n <= 0 
            y < 0
            g(0) != 0
    TypeError 
        if n is not an integer.
    RuntimeError 
        if a suitable approximation is not found by max_iter iterations.
    
    Big-O Complexity 
    -------------
    O(log_2 n) assuming inexpensive g because binary search cuts search window in half with each iteration
    '''
    # input validation 
    _input_validation_compInvg(n, y, g, float_tolerance)
    
    # We first identify a 1-wide interval within which g first exceeds y
    j = _find_window_compInvg(n, y, g, max_iter, init_guess, float_tolerance)
    
    # binary search
    left = j*n 
    if g(left/n) > y + float_tolerance:
        return Fraction(left, n) 
    right = (j + 1)*n 
    
    while left <= right:
        k = (left + right)//2 
        
        if g(k/n) <= y + float_tolerance and g((k + 1)/n) > y + float_tolerance: 
            return Fraction(k + 1, n)
        
        elif g(k/n) <= y + float_tolerance and g((k + 1)/n) <= y + float_tolerance:
            left = k + 1
        
        elif g(k/n) > y + float_tolerance:
            right = k - 1 
    
    return Fraction(left, n)

# ALGORITHM 1.2
def DistSpec(matrix : Callable[[int, int], complex], n : int, z : Union[complex, tuple[Fraction, Fraction]], f : Callable[[int], int], fn : int = None, max_iter : int = config.max_iter, float_tolerance : float = config.float_tolerance) -> Fraction:
    '''
    Approximate norm(R(z, A))^(-1) with mesh size 1/n given dispersion f
        
    Parameters 
    -------------
    matrix : Callable[[int, int], complex]
        function N^2 -> C representing a closed infinite matrix. 
    
    In the following, A = (matrix(i, j))
    n : int 
        size of mesh, must satisfy n > 0
    z : complex or tuple[Fraction, Fraction]
        z in norm(R(z, A))^(-1). tuple[Fraction, Fraction] allows more precise specification of value in other methods.
    f : Callable[[int], int]
        a dispersion control for the matrix, accepting ints and giving ints. Must satisfy f(n) >= n + 1 and be increasing.
    fn : int 
        the value of f(n). We allow pre-computation of fn in case DistSpec(_slow) is involved in a loop and f(n) is re-computed repeatedly.
    max_iter : int 
        maximum number of iterations to find k/n before termination. Defaults to module default (default 0)
    float_tolerance : float 
        error margin for floating point calculations. Defaults to module default (default 1e-10) 
      
    
    Returns
    -------------
    Fraction
        rational approximation for norm(R(z, A))^(-1)
    
    Raises 
    -------------
    TypeError
        if f(n) is not an integer
    ValueError 
        if f(n) does not satisfy f(n) >= n + 1: f is not a valid dispersion bound 
    RuntimeError
        if a suitable approximation is not found in max_iter iterations
    
    Big-O Complexity 
    -------------
    O(n^3 + n) 
    
    1. Computing eigenvalues is O(n^3)
    2. Finding the minimum of these eigenvalues is O(n) + O(n) = O(n)
    3. Since l^2/n^2 is approximately min_eigval, l is approximately n * sqrt(min_eigval) = O(n) 
    4. Hence we will then do O(n) iterations to find l 
    5. Hence the final complexity is O(n^3) + O(n) + O(n) = O(n^3 + n)
    '''
    fn = f(n) if fn is None else fn # pre-compute f(n) in case it is expensive
    
    # check f 
    _validate_f(f, n, fn)
    
    if isinstance(z, tuple[Fraction, Fraction]):
        z = (z[0]+z[1]j)
        
    # prepare matrices 
    B = _generate_matrix(matrix, fn, n, z) # (A - z I)(1 : f(n))(1 : n)
    C = np.conjugate(_generate_matrix(matrix, n, fn, z)).T # (A - z I)*(1 : f(n))(1 : n), same as ((A - z I)(1 : n)(1 : f(n)))*
    S = np.matmul(np.conjugate(B).T, B) # S = B* mul B
    S_size = S.shape[0] # get size of S to identify suitable identity matrix 
    id_S = np.identity(S_size)
    T = np.matmul(np.conjugate(C).T, C) # T = C* mul C
    T_size = T.shape[0] # get size of T to identify suitable identity matrix 
    id_T = np.identity(T_size)
    
    eigvals_S = np.linalg.eigvalsh(S) 
    eigvals_T = np.linalg.eigvalsh(T)
    
    min_eigvals_S = min(eigvals_S)
    min_eigvals_T = min(eigvals_T)
    
    min_eigval = min(min_eigvals_S, min_eigvals_T)
    
    threshold = min_eigval - float_tolerance 
    
    if threshold <= 0:
        return Fraction(1, n) 
    
    else:
        l = math.ceil(n * sqrt(threshold))
        return Fraction(l, n)

    if l == max_iter:
        raise RuntimeError(f"max_iter ({max_iter}) exceeded")
    
    return Fraction(l, n) # using fraction to avoid floating point errors

def _input_validation_generate_grid(n : int) -> None:
    '''
    Input validation for generate_grid and generate_grid_slow. Not intended to be called directly. 
    '''
    if not isinstance(n, int):
        raise TypeError("Grid size n is not an int")
    
    if n <= 0:
        raise ValueError("Grid size n is non-positive")

def generate_grid_slow(n : int) -> list[complex]:
    '''
    Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes. 
    
    _slow: Brute force method checking each candidate (x, y) individually. Not called by any other method. 
    
    Parameters
    -------------
    n : int 
        mesh size for grid
    
    Returns 
    -------------
    list[tuple[Fraction, Fraction]]
        list of tuples of Fractions corresponding to complex numbers in Grid(n)  
    
    Raises 
    -------------
    TypeError: 
        if n is not an integer
    ValueError:
        if n <= 0
    Big-O Complexity 
    -------------
    O(n^4) - n^2 values of x, and n^2 values of y for each x. 
    '''
    # input validation
    _input_validation_generate_grid(n)
    # return 
    return [
        (Fraction(x, n), Fraction(y, n))
        for x in range(-n*n, n*n + 1) 
        for y in range(-n*n, n*n + 1) 
        if x*x + y*y <= n**4
    ]

def generate_grid(n : int) -> list[tuple[Fraction, Fraction]]:
    '''
    Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes (represented by tuples of Fractions).
    
    not _slow: Given an x, (x, y) \in Grid(n) if and only if y^2 <= n^4 - x^2. That is, if and only if |y| <= floor(sqrt(n^4 - x^2)) := y_max. Hence we enumerate up to this y_max.
    
    Parameters
    -------------
    n : int 
        mesh size for grid
    
    Returns 
    -------------
    list[tuple[Fraction, Fraction]]
        list of tuples of Fractions corresponding to complex numbers in Grid(n) 
    
    Raises 
    -------------
    TypeError: 
        if n is not an integer
    ValueError:
        if n <= 0
    
    Big-O Complexity 
    -------------
    O(n^4), but slightly (typically performs ~79% as many calcluations) faster because of narrowed search range. 
    '''
    # input validation 
    _input_validation_generate_grid(n)
    
    # prepare for loop  
    n2 = n*n # pre-compute n^2 so we don't have to re-compute it every loop
    n4 = n2*n2 # pre-compute n^4 to avoid re-computation
    
    return [
        (Fraction(x, n), Fraction(y, n))
        for x in range(-n2, n2 + 1)
        for y in range(-isqrt(n4 - x*x), isqrt(n4 - x*x) + 1)
    ] # note that x^2 + y^2 <= n^4 is equivalent to |y| <= sqrt(n^4 - x^2), for each x we simply include the y with |y| <= sqrt(n^4 - x^2). 

def intersect_grid_with_ball(n : int, rad : Fraction, centre : tuple[Fraction, Fraction]) -> list[tuple[Fraction, Fraction]]:
    '''
    Generates Grid(n) \cap B_r(z), where r = rad and z = centre. 
    
    Parameters
    -------------
    n : int 
        mesh size for grid
    rad : Fraction 
        radius of ball
    centre : tuple[Fraction, Fraction]
        centre of ball 

    Returns
    -------------
    list[tuple[Fraction, Fraction]]
        list of tuples of Fractions corresponding to complex numbers in Grid(n) \cap B_r(z), where B_r(z) is the closed ball with radius r and centre z.
    
    Raises 
    -------------
    TypeError: 
        if n is not an integer (propagated from generate_grid)
    ValueError:
        if n <= 0 (propagated from generate_grid) 
        if r < 0 (cannot have ball with negative radius. Non-zero radius is trivial (contains only z) but accepted.
    '''
    grid = generate_grid(n)
    
    if rad < 0:
        raise ValueError("rad is negative.")
    
    r2 = rad*rad # pre-compute r^2 so it doesn't get re-computed for every w_j in grid. 
    return [
        w_j 
        for w_j in grid 
        if (w_j[1] - centre[1])*(w_j[1] - centre[1]) + (w_j[0] - centre[0])*(w_j[0] - centre[0]) <= r2 # if |w_j - z|^2 <= r^2. We check squares so we are comparing rational numbers rather than floats.
    ]

# ALGORITHM 1.3
def CompSpecUB(matrix : Callable[[int, int], complex], n : int, g : Callable[[float], float], f : Callable[[int], int], fn : int = None, c : Callable[[int], Fraction], c_n : Fraction = None) -> tuple[list[tuple[Fraction, Fraction]], Fraction]:
    '''
    Computes a tuple consisting of an approximation to the spectrum of the matrix input, as well as a bound on the error of this approximation. 
    
    Parameters
    -------------
    matrix : Callable[[int, int], complex]
        function N^2 -> C representing a closed infinite matrix.
    n : int 
        degree of approximation 
    g : Callable[[float], float] 
        increasing function g : R_+ -> R_+ representing resolvent control. Must satisfy g(0) = 0, g(x) <= x and be monotone increasing. That g(x) <= x or g is monotone is not checked.
    f : Callable[[int], int]
        a dispersion control for the matrix, accepting ints and giving ints. Must satisfy f(n) >= n + 1 and be increasing.
    fn : int 
        allows for the value of f(n) to be pre-loaded, for example if this method is to be called in a loop and computing f is expensive. It is never checked that fn = f(n). 
    c : Callable[[int], Fraction]
        a sequence satisfying D_(f, n)(A) <= c_n 
    c_n : Fraction 
        allows for the value of c_n to be pre-loaded, for example if this method is to be called in a loop and computing c is expensive. It is never checked whether c_n = c(n).
    
    Returns
    -------------
    tuple[list[tuple[Fraction, Fraction], float]
        first element of tuple represents an approximation to the spectrum of the matrix, the second is a float giving the approximation error. 
    
    Raises 
    -------------
    TypeError: 
        if n is not an integer (propagated from generate_grid)
        if f(n) is not an integer (propagated from _validate_f)
    ValueError:
        if n <= 0 (propagated from generate_grid)
        if f(n) < n + 1 (propagated from _validate_f)
    '''
    
    fn = f(n) if fn is None else fn # pre-compute f(n) to avoid re-computation in loop if it is not passed as pre-computed parameter
    _validate_f(f, n, fn) # check f
    c_n = c(n) if c_n is None else c_n # pre-compute c_n to avoid re-computation in loop if it is not passed as pre-computed parameter
    grid = generate_grid(n)
    cur_min = None 
    E_n = 0
    Gamma_n = []
    for z in grid:
        reset_gamma = False
        Fz = DistSpec(matrix, n, z, f, fn)
        W_z = []
        
        if Fz*(z[0]*z[0] + z[1]*z[1] + 1) <= 1:
            for w_j in intersect_grid_with_ball(CompInvg(n, Fz, g), z):
                F_j = DistSpec(matrix, n, w_j, f, fn)
                cur_min = F_j if cur_min is None else cur_min # if cur_min is still unspecified, take it to be F_j
                
                if cur_min == F_j: 
                    W_z.append(w_j) 
                
                elif cur_min < F_j:
                    cur_min = F_j 
                    W_z = [w_j] 
                    # we've found a new minimizer so the current Gamma must be cleared. In the worst case a new minimizer will be discovered every loop, so this avoids continuously rewriting Gamma_n. 
                    reset_gamma = True 
                # else F_j is not a minimizer and we continue our search
        if reset_gamma:
            Gamma_n = W_z 
        
        else:
            Gamma_n.extend(W_z)
    for z in Gamma_n:
        # since Gamma_n may be very large, we avoid creating a new list.  
        E_n = max(E_n, CompInvg(n, DistSpec(matrix, n, z, fn) + c_n, g))
    
    return Gamma_n, E_n

# ALGORITHM 2
def PseudoSpecUB(matrix : Callable[[int, int], complex], eps : Fraction, n : int, f : Callable[[int], int], c : Callable[[int], Fraction], fn : int = None, c_n : Fraction = None):
    '''
    Computes an nth order approximation to the epsilon-pseudospectrum.
    
    Parameters
    -------------
    matrix : Callable[[int, int], complex]
        function N^2 -> C representing a closed infinite matrix.
    eps : Fraction 
        the epsilon for which we will compute the epsilon-pseudospectrum
    n : int 
        degree of approximation
    f : Callable[[int], int]
        a dispersion control for the matrix, accepting ints and giving ints. Must satisfy f(n) >= n + 1 and be increasing.
    c : Callable[[int], Fraction]
        a sequence satisfying D_(f, n)(A) <= c_n 
    fn : int 
        allows for the value of f(n) to be pre-loaded, for example if this method is to be called in a loop and computing f is expensive. It is never checked that fn = f(n).
    c_n : Fraction 
        allows for the value of c_n to be pre-loaded, for example if this method is to be called in a loop and computing c is expensive. It is never checked whether c_n = c(n).
    
    Raises 
    -------------
    TypeError: 
        if n is not an integer (propagated from generate_grid)
        if f(n) is not an integer (propagated from _validate_f)
    ValueError:
        if n <= 0 (propagated from generate_grid)
        if f(n) < n + 1 (propagated from _validate_f)
        if eps <= 0 (propagated from _validate_eps)
    '''
    eps = _validate_eps(eps)
    grid = generate_grid(n)
    fn = f(n) if fn is None else fn # pre-compute f(n) to save DistSpec the trouble of re-computing it every time it is called 
    _validate_f(f, n, fn) # check f
    c_n = c(n) if c_n is None else c_n # pre-compute c_n
    return [
        z 
        for z in grid 
        if DistSpec(matrix, n, z, f, fn) + c_n < eps
    ]

# ALGORITHM 3.1
def TestSpec(n1 : int, n2 : int, K_n2 : list[float], gamma_n1 : Callable[[complex], float], float_tolerance : float = config.float_tolerance) -> bool:
    '''
    Given (n_1, n_2), an approximation K_n2 to a compact set K, an approximation gamma_n1(z) to gamma(z, A), return an approximation to the truth value of K \cap spec(A) \ne emptyset.
    
    Validity of approximation depends on the choice of approximation K_n2 and gamma_n1. 
    
    Parameters
    -------------
    n1 : int 
        Degree of approximation corresponding to the resolvent gamma(z, A). 
    n2 : int 
        Degree of approximation corresponding to the compact set K. 
    K_n2 : list[float]
        A finite list of complex numbers approximating K in Hausdorff metric.
    gamma_n1 : Callable[[complex], float]
        An approximation to gamma(z, A) 
    float_tolerance : float 
        Tolerance applied for floating point error. Defaults to config.float_tolerance.
    
    Raises 
    -------------
    TypeError
        If n1 or n2 is not an integer. Propagated from _validate_order_approx.
        If float_tolerance is not a float or Fraction. Propagated from _validate_float_tolerance.
    ValueError
        If n1 or n2 is negative. Propagated from _validate_order_approx.
        If float_tolerance is non-positive. Propagated from _validate_float_tolerance.
    '''
    _validate_TestSpec(n1, n2)
    _validate_float_tolerance(float_tolerance)
    
    for z in K_n2:
        if (1 << n2) * gamma_n1(z) + float_tolerance < 1: # note that 1 << n is much cheaper than 2**n for large n. multiply through by 2^(n_2) to avoid comparing small floats.
            return True
    
    return False

# ALGORITHM 3.2
def TestPseudoSpec(n1 : int, n2 : int, K_n2 : list[complex], gamma_n1 : Callable[[complex], float], eps : Union[float, Fraction], float_tolerance : float = config.float_tolerance) -> bool:
    '''
    Given (n_1, n_2), an approximation K_n2 to a compact set K, an approximation gamma_n1(z) to gamma(z, A), return an approximation to the truth value of K \cap spec_eps(A) \ne \emptyset.
    
    Validity of approximation depends on the choice of approximation K_n2 and gamma_n1. 
    
    Parameters
    -------------
    n1 : int 
        Degree of approximation corresponding to the resolvent gamma(z, A). 
    n2 : int 
        Degree of approximation corresponding to the compact set K. 
    K_n2 : list[float]
        A finite list of complex numbers approximating K in Hausdorff metric.
    gamma_n1 : Callable[[complex], float]
        An approximation to gamma(z, A) 
    eps : Union[float, Fraction]
        The epsilon in the claim K \cap spec_eps(A) \ne \emptyset. 
    float_tolerance : float 
        Tolerance applied for floating point error. Defaults to config.float_tolerance.
    
    Raises 
    -------------
    TypeError
        If n1 or n2 is not an integer. Propagated from _validate_order_approx.
        If float_tolerance is not a float or Fraction. Propagated from _validate_float_tolerance.
    ValueError
        If n1 or n2 is negative. Propagated from _validate_order_approx.
        If float_tolerance is non-positive. Propagated from _validate_float_tolerance.
    '''
    _validate_order_approx(n1, n2)
    _validate_float_tolerance(float_tolerance)
    eps = _validate_eps(eps)
    
    for z in K_n2 
        if (1 << n2) * gamma_n1(z) + float_tolerance < 1 + eps:
            return True 
    
    return False

# ALGORITHM 4
def SpecGap(n1 : int, n2 : int, projected_matrix : np.array, float_tolerance : Union[float, Fraction] = config.float_tolerance) -> bool:
    '''
    Returns an approximation to the truth value of "is the spectrum of A gapped?". A is a Hermitian operator.
    
    We say that the spectrum of A is gapped if the minimum of spec(A) is an isolated eigenvalue with multiplicity 1.
    
    Parameters
    -------------
    n1 : int 
        The first order of the approximation. 
    n2 : int 
        The second order of the approximation. 
    projected_matrix : np.array 
        The matrix P_{n1} A P_{n1} given as an np.array, where P_{n1} denotes the projection onto the first n basis elements.
    float_tolerance : float 
        error margin for floating point calculations. Defaults to module default (default 1e-10)
    
    Returns 
    -------------
    bool 
        An approximation to the truth value of "A is gapped". 
    
    Raises 
    -------------
    TypeError
        If n1 or n2 is not an integer. Propagated from _validate_order_approx.
        If float_tolerance is not a float or Fraction. Propagated from _validate_float_tolerance.
    ValueError
        If projected_matrix is not Hermitian. Propagated from _validate_matrix_hermitian.
        If n1 or n2 is negative. Propagated from _validate_order_approx.
        If float_tolerance is non-positive. Propagated from _validate_float_tolerance.
    '''
    _validate_matrix_hermitian(projected_matrix)
    _validate_order_approx(n1, n2)
    _validate_float_tolerance(float_tolerance)
   
    if n1 == 1:
        return True
    
    result = False
    for k in range(2, n1 + 1):
        trunc = projected_matrix[:k, :k] # compute P_k A P_k
        eigvals = sorted(np.linalg.eigvalsh(trunc)) 
        gap = eigvals[1] - eigvals[0] # ie. l_k = mu_2^(k) - mu_1^(k)
        # write J_(n_2)^1 = [0, 1/(2n_2)] and J_(n_2)^2 = (1/n_2, inf) as in the paper. 
        if gap*(2*n2) <= 1 + float_tolerance: # ie. l_k \in J_(n_2)^1
            result = False 
        if gap*(n2) > 1 + float_tolerance: # ie. l_k \in J_(n_2)^2
            result = True 
    # as we exit the loop, the present value of result will reflect the maximum k <= n_1
    # as in the paper, if this k has l_k \in J_(n_2)^1, we will output False, and otherwise we will output True. 
    # if there is no k such that l_k \in J_(n_1)^1 \cup J_(n_2)^2, then neither of the if conditions will be satisfied and the initial assignment of False will persist.
    return result

