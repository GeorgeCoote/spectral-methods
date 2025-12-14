#SOURCE: https://www.damtp.cam.ac.uk/user/mjc249/pdfs/JEMS_foundations_colbrook.pdf The foundations of spectral computations via the Solvability Complexity Index hierarchy by Matthew J. Colbrook, Anders C. Hansen (2022)

from fractions import Fraction
from collections.abc import Callable
from math import isqrt
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

        zero_tolerance : float 
            Resolvent bounds g may be computed inexactly, yet are still expected to satisfy g(0.0) = 0.0. Implementing a direct inequality check may give false
            negatives, hence we consider g(0) = 0 validated if g(0) < zero_tolerance. Should be optimized based on the level of precision expected.
            

        init_guess : int
            Where an integer is being approximated, init_guess is used as its default value. If the integer is expected to be large, one may consider using a
            better initial guess.

        int_allowed : list[dtype]
            List of allowed data types for integers. 

        float_allowed : list[dtype]
            

        Returns
        -------------
        None

        Raises
        -------------
        None, hopefully
        '''
        self.max_iter = 10_000_000
        self.zero_tolerance = 1e-10
        self.init_guess = 0
        self.allowed_types = {
            'max_iter': [int],
            'zero_tolerance': [float],
            'init_guess': [int]
        }
        self.int_allowed = [int]
        self.float_allowed = [float]
    
    def update(self, **kwargs) -> None:
        '''Updates module constants

        Parameters
        -------------
        max_iter : int
            Number of iterations to attempt before terminating method. High iteration counts can indicate unoptimal parameters or a faulty method given as input.

            For example, a given resolvent bound may loop infinitely. 

        zero_tolerance : float 
            Resolvent bounds g may be computed inexactly, yet are still expected to satisfy g(0.0) = 0.0. Implementing a direct inequality check may give false
            negatives, hence we consider g(0) = 0 validated if g(0) < zero_tolerance. Should be optimized based on the level of precision expected.

        init_guess : int
            Where an integer is being approximated, init_guess is used as its default value. If the integer is expected to be large, one may consider using a
            better initial guess.
        '''
        for key, value in kwargs.items():
            if hasattr(self, key):
                if type_check(key, value):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"{type(key)} not an allowed type for {key}. Allowed types are {allowed_types(key)}")
            else:
                raise ValueError(f"Unknown parameter: {key}")
    def reset(self) -> None:
        '''Reset module constants to default value'''
        self.__init__()
    def get_values(self) -> dict:
        return {
            'max_iter': self.max_iter,
            'zero_tolerance': self.zero_tolerance,
            'init_guess': self.init_guess,
            'int_allowed': self.int_allowed,
            'float_allowed': self.float_allowed
        }
    
    def type_check(self, key, value):
        return type(value) in allowed_types(key)

config = Config() # init config

#general helper functions 
def _generate_matrix(matrix:Callable[[int, int], float], m, n, z = 0) -> np.array:
    '''
    Method to convert a matrix as a callable into a numpy array by vectorizing the matrix. Cheaper than creating a list and then converting to numpy. Not intended to be called directly. 
    '''
    vectorized_matrix = np.vectorize(matrix) 
    i_grid, j_grid = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    output_matrix = vectorized_matrix(i_grid, j_grid) - z
    return output_matrix

# TODO: method to change defaults 
    
    
# CompInvg

def _input_validation_compInvg(n: int, y: float, g:Callable[[float], float]) -> None:
    '''
    Input validation for CompInvg_slow and CompInvg. Not intended to be called directly. 
    '''
    if not isinstance(n, int): 
        raise TypeError("Precision n must be an int and not float") 
    if n <= 0:
        raise ValueError("Precision n must be positive")  
    if y < 0:
        raise ValueError("y in g^(-1)(y) must be non-negative") 
    if abs(g(0.0)) > DEFAULT_zero_tolerance:
        raise ValueError("We must have g(0) = 0, with g our resolvent bound. This g(0) falls out of floating point tolerance.")

def CompInvg_slow(n: int, y: float, g:Callable[[float], float], max_iter:int = config.max_iter, init_guess:int = config.init_guess) -> Fraction:
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
    _input_validation_compInvg(n, y, g)
    # We first identify a 1-wide interval within which g first exceeds y
    j = init_guess
    while g(j + 1) <= y and j < max_iter:
        j += 1
    if j == max_iter:
        raise RuntimeError(f"max_iter ({max_iter}) exceeded. Last checked g({max_iter}) = {g(max_iter)}. Consider optimizing init_guess.")
    # once we've exited this loop, we know g(j + 1) > y and g(j) <= y. Hence g^(-1)(y) \in [j, j + 1) = [(j*n)/n, (j*(n + 1))/n)
    for k in range(j*n, (j + 1)*n):
        if g(k/n) > y:
            return Fraction(k, n) # using fraction to avoid floating point errors

#DistSpec 
def DistSpec_slow(matrix:Callable[[int, int], complex], n:int, z:complex, f:Callable[[int], int], max_iter:int = config.max_iter) -> Fraction:
    '''
    Approximate norm(R(z, A))^(-1) with mesh size 1/n given dispersion f
    
    _slow: Brute force method, checks each l individually and computes all eigenvalues before concluding on positive definiteness.
    
    Parameters 
    -------------
    matrix : collections.abc.Callable 
        function N^2 -> C representing a closed infinite matrix. 
    
    In the following, A = (matrix(i, j))
    n : int 
        size of mesh, must satisfy n > 0
    z : complex 
        z in norm(R(z, A))^(-1)
    f : collections.abc.Callable[[int], int]
        a dispersion control for the matrix, accepting ints and giving ints
    max_iter : int 
        maximum number of iterations to find k/n before termination
    
    Returns
    -------------
    Fraction
        rational approximation for norm(R(z, A))^(-1)
    
    Raises 
    -------------
    TypeError
        if f(n) is not an integer
    ValueError 
        if f(n) does not satisfy f(n) >= n: f is not a valid dispersion bound 
    RuntimeError
        if a suitable approximation is not found in max_iter iterations
    
    Big-O Complexity 
    -------------
    O(f(n)*n^2 + max_iter*n^3) - since max_iter dominates f(n) this will scale more like O(max_iter*n^3). 
    
    first term: O(f(n)*n) for building the matrices, O(f(n)*n^2) from matrix multiplication.
    
    second term: eigenvalue search is O(n^3), we do this up to max_iter times.
    '''
    fn = f(n) # pre-compute f(n) in case it is expensive
    if not (isinstance(fn, int)): #check if f(n) is an integer 
        raise TypeError(f"f(n) ({fn}) is not an int")
    if not fn >= n: # check if f(n) >= n
        raise ValueError(f"f(n) ({fn}) is not >= n")
    B = _generate_matrix(matrix, fn, n, z) # (A - z I)(1 : f(n))(1 : n)
    C = np.conjugate(_generate_matrix(matrix, n, fn, z)).T # (A - z I)*(1 : f(n))(1 : n), same as ((A - z I)(1 : n)(1 : f(n)))*
    S = np.matmul(np.conjugate(B).T, B) # S = B*B
    S_size = S.shape[0] # get size of S to identify suitable identity matrix 
    id_S = np.identity(S_size)
    T = np.matmul(np.conjugate(C).T, C) # T = C*C
    T_size = T.shape[0] # get size of T to identify suitable identity matrix 
    id_T = np.identity(T_size)
    v = True
    l = 1
    while v and l < max_iter:
        l += 1
        l2 = l*l  
        n2 = n*n
        p = np.all(np.linalg.eigvalsh(S - (l2/n2)*id_S) > 0) # check whether S - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
        q = np.all(np.linalg.eigvalsh(T - (l2/n2)*id_T) > 0) # check whether T - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
        v = p and q
    if l == max_iter:
        raise RuntimeError(f"max_iter ({max_iter}) exceeded")
    return Fraction(l, n) # using fraction to avoid floating point errors

# grid generation 
def _input_validation_generate_grid(n:int) -> None:
    '''
    Input validation for generate_grid and generate_grid_slow. Not intended to be called directly. 
    '''
    if not isinstance(n, int):
        raise TypeError("Grid size n is not an int")
    if n <= 0:
        raise ValueError("Grid size n is non-positive")

def generate_grid_slow(n:int) -> list[complex]:
    '''
    Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes . 
    
    _slow: Brute force method checking each candidate (x, y) individually. 
    
    Parameters
    -------------
    n : int 
        mesh size for grid
    
    Returns 
    -------------
    list[complex]
        List of complex numbers corresponding to Grid(n) 
    
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
    return [complex(x, y)/n for x in range(-n*n, n*n + 1) for y in range(-n*n, n*n + 1) if x*x + y*y <= n**4]

def generate_grid(n:int) -> list[complex]:
    '''
    Generates 1/n (Z + i Z) \cap B_n(0) = Grid(n) as a list of complexes.
    
    not _slow: Given an x, (x, y) \in Grid(n) if and only if y^2 <= n^4 - x^2. That is, if and only if |y| <= floor(sqrt(n^4 - x^2)) := y_max. Hence we enumerate up to this y_max.
    
    Parameters
    -------------
    n : int 
        mesh size for grid
    
    Returns 
    -------------
    list[complex]
        List of complex numbers corresponding to Grid(n) 
    
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
    #method 
    grid = [] #init empty grid
    n2 = n*n #pre-compute n^2 so we don't have to re-compute it every loop
    n4 = n2*n2 # pre-compute n^4 to avoid re-computation
    for x in range(-n2, n2 + 1): #iterate over x
        #as in docstring, y_max cannot be bigger than this and moreover every y in this range corresponds to a point in Grid(n) 
        max_y = isqrt(n4 - x*x) 
        for y in range(-max_y, max_y + 1):
            grid.append(complex(x/n, y/n))
    return grid

# CompSpecUB
