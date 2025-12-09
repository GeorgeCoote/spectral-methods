#SOURCE: https://www.damtp.cam.ac.uk/user/mjc249/pdfs/JEMS_foundations_colbrook.pdf The foundations of spectral computations via the Solvability Complexity Index hierarchy by Matthew J. Colbrook, Anders C. Hansen (2022)

from fractions import Fraction
from collections.abc import Callable
import numpy as np
class JEMScolbrook:
    # CompInvg
    def _type_validation_compInvg(self, n: int, y: float, g:Callable[[float], float]) -> None:
        if not isinstance(n, int): 
            raise TypeError("n must be an int and not float") 
        if n <= 0:
            raise ValueError("n must be positive")  
        if y < 0:
            raise ValueError("y must be non-negative") 
        if g(0.0) != 0.0:
            raise ValueError("We must have g(0) = 0")
    def CompInvg_slow(self, n: int, y: float, g:Callable[[float], float], max_iter = 10**7, init_guess = 0) -> Fraction:
        '''
        Approximate g^(-1)(y) using a discrete mesh of size 1/n. Specifically, we find the least k such that g(k/n) > y and hence give an approximation to g^(-1)(y) to precision 1/n. 

        _slow: Does not use binary search to find approximation, and hence is presented for theoretical interest.
        
        Parameters
        -------------
        n : int 
            size of mesh, must satisfy n > 0
        y : float 
            input for which we want to approximate g^(-1)(y)
        g : collections.abc.Callable[[float], float]
            increasing function g : R_+ -> R_+ representing resolvent control. Must satisfy g(0) = 0, g(x) <= x and be monotone increasing. That g(x) <= x or g is monotone is not checked.
        max_iter : int 
            maximum number of iterations to find k/n before termination
        init_guess : int 
            initial guess for k.
        
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
        '''
        # input validation 
        self._type_validation_compInvg(n, y, g)
        # We first identify a 1-wide interval within which g first exceeds y
        j = init_guess
        while g(j + 1) <= y:
            j += 1
            if j == max_iter:
                raise RuntimeError(f"max_iter ({max_iter}) exceeded")
        # once we've exited this loop, we know g(j + 1) > y and g(j) <= y. Hence g^(-1)(y) \in [j, j + 1) = [(j*n)/n, (j*(n + 1))/n)
        for k in range(j*n, (j + 1)*n):
            if g(k/n) > y:
                return Fraction(k, n) # using fraction to avoid floating point errors
    
    #DistSpec 
    def DistSpec_slow(self, matrix:Callable[[int, int], complex], n:int, z:complex, f:Callable[[int], int], max_iter = 10**7) -> Fraction:
        '''
        Approximate norm(R(z, A))^(-1) with mesh size 1/n given dispersion f
        
        _slow: Does not use binary search to find approximation and computes all eigenvalues to check positive definiteness, and hence is presented for theoretical interest.
        
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
        '''
        fn = f(n) # pre-compute f(n) in case it is expensive
        if not (isinstance(fn, int)): #check if f(n) is an integer 
            raise TypeError(f"f(n) ({fn}) is not an int")
        if not fn >= n: # check if f(n) >= n
            raise ValueError(f"f(n) ({fn}) is not >= n")
        B = np.array([[matrix(i, j) - z for j in range(n)] for i in range(fn)]) # (A - z I)(1 : f(n))(1 : n)
        C = np.array([[matrix(j, i) - z for j in range(n)] for i in range(fn)]) # (A - z I)*(1 : f(n))(1 : n)
        S = np.matmul(np.conjugate(B).T, B) # S = B*B
        S_size = S.shape[0] # get size of S to identify suitable identity matrix 
        T = np.matmul(np.conjugate(C).T, C) # T = C*C 
        T_size = T.shape[0] # get size of T to identify suitable identity matrix 
        v = True
        l = 1
        while v and l < max_iter:
            l += 1
            p = np.all(np.linalg.eigvalsh(S - ((l*l)/(n*n))*np.identity(S_size)) > 0) # check whether S - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
            q = np.all(np.linalg.eigvalsh(T - ((l*l)/(n*n))*np.identity(T_size)) > 0) # check whether T - l^2/n^2 I is positive definite. This represents an upper bound on distance to the spectrum
            v = p and q
        if l == max_iter:
            raise RuntimeError(f"max_iter ({max_iter}) exceeded")
        return Fraction(l, n) # using fraction to avoid floating point errors