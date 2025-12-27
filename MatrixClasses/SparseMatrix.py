from fractions import Fraction
from typing import Union

class SparseMatrix:
    '''
    Class representing a "sparse matrix". 
    
    Informally, a sparse matrix is a matrix whose entries are "mostly" zero. 
    
    Formally, we say a sparse matrix has only finitely many non-zero values down any given column and across any given row. 
    
    This matrix will be represented by a map N^2 -> C. 
    
    Furthermore, we have access to a computable function f which instructs us on the location of the non-zero entries.
    '''
    def __init__(self, entries : Callable[[int, int], Union[float, int, complex, Fraction]], f : Callable[[int], int], tolerance : Union[float, Fraction] = 1e-16) -> None:
        '''
        Initialises the matrix A by setting object attributes.
        
        We write A = (a_ij), where a_ij is the matrix element on the ith row, jth column. 
        
        Parameters
        -------------
        entries : Callable[[int, int], Union[float, int, complex, Fraction]]
            A callable giving the entries of A. Given (i, j) as input, the callable should return the matrix element a_ij. 
        
        f : Callable[[int], int]
            A callable satisfying a_ij = 0 for i > f(j) and j > f(i). This condition is assumed to be user-verified and not checked.
            
            This tells us that on the ith row, there are no non-zero entries beyond column f(i), meaning that we only have to read up to a_(i f(i)) for calculations.
            
            It also tells us that on the jth column, there are no non-zero entries beyond row f(j), meaning that we only hae to read up to a_(f(j) j).
            
            While the matrix is infinite, f allows us to know where the non-zero matrix elements are, allowing the exploitation of sparsity.
            
        tolerance : Union[float, Fraction]
            Where a comparison with a float is required, this argument gives the offset used to account for floating point imprecision. 
            
            If abs(c) < tolerance, we will treat c as zero.
            
            Default argument is 1e-16 
        
        Returns
        -------------
        None 
        
        Raises
        -------------
        TypeError 
            If tolerance is not of type float or Fraction.  
        '''
        if not isinstance(tolerance, (float, Fraction)):
            raise TypeError("Tolerance must be float or Fraction")
        self.entries = entries 
        self.f = f
        self.tolerance = tolerance 
    
    def __call__(self, i : int, j : int) -> Union[float, int, complex, Fraction]:
        '''
        Allows a SparseMatrix to be used as a callable. 
        
        If A = (a_ij) is a SparseMatrix, then we now have A(i, j) = a_ij. 
        
        Parameters 
        i : int
            Gives the row number of the requested element.
        j : int
            Gives the column number of the requested element.

        Returns
        -------------
        Union[float, int, complex, Fraction]
            The desired matrix element. May be a float, int, complex or Fraction.
        '''
        return 0 if (j > self.f(i) or i > self.f(j)) else self.entries(i, j)
    
    def __add__(self, B : 'SparseMatrix') -> 'SparseMatrix':
        '''
        Allows addition of two SparseMatrix objects by overloading the + operator. 

        In the following, we write A = (a_ij) = self for simplicity.

        Given a SparseMatrix B = (b_ij), we return A + B = (a_ij + b_ij).

        Parameters
        -------------
        B : SparseMatrix
            The B in A + B.

        Returns
        -------------
        SparseMatrix
            A + B as a SparseMatrix.

        Raises
        -------------
        TypeError
            If B is not a SparseMatrix. 
        '''
        A = self 
        
        if not isinstance(B, SparseMatrix):
            raise TypeError("Cannot add SparseMatrix with object not of SparseMatrix type.")
        
        new_entries = lambda i, j : A.entries(i, j) + B.entries(i, j)
        
        # Note that we have a_ij = 0 for i > A.f(j) or j > A.f(i) and b_ij = 0 for i > B.f(j) or j > B.f(i) 
        
        # Hence, if g = max(A.f, B.f) = new_f, we have a_ij = 0 and b_ij = 0 (and so a_ij + b_ij = 0) if i > g(j) or j > g(i).
        
        # So new_f = max(A.f, B.f) is used as the f for A + B.
        
        new_f = lambda x : max(A.f(x), B.f(x))
        new_tolerance = min(A.tolerance, B.tolerance)
        
        return SparseMatrix(new_entries, new_f, new_tolerance)
    
    def __rmul__(self, c : Union[float, int, complex, Fraction) -> 'SparseMatrix':
        A = self
        
        if not isinstance(c, (float, int, complex, Fraction)):
            raise TypeError("Cannot multiply FiniteSparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction")
        
        if abs(c) < A.tolerance:
            return SparseMatrix(lambda x : 0, lambda x : 0, A.tolerance)
        
        else:
            new_entries = lambda i, j : c*A.entries(i, j)
            return SparseMatrix(new_entries, A.f, A.tolerance)
    
    def __matmul__(self, B : 'SparseMatrix') -> 'SparseMatrix':
        pass 
    
    def __sub__(self, B : 'SparseMatrix') -> 'SparseMatrix':
        A = self 
        
        if not isinstance(B, FiniteSparseMatrix):
            raise TypeError("Cannot add SparseMatrix with object not of SparseMatrix type.")
        
        new_entries = lambda i, j : A.entries(i, j) - B.entries(i, j)
        new_f = lambda x : max(A.f(x), B.f(x))
        new_tolerance = min(A.tolerance, B.tolerance)
        
        return SparseMatrix(new_entries, new_f, new_tolerance)
    
    def __repr__(self) -> str:
        return f"SparseMatrix({self.entries}, {self.f}, {self.tolerance})"
    
    def get_entries(self) -> Callable[[int, int], Union[float, int, complex, Fraction]]:
        return self.entries 
    
    def set_entries(self, entries : Callable[[int, int], Union[float, int, complex, Fraction]]) -> None:
        self.entries = entries 
    
    def get_f(self) -> Callable[[int], int]:
        return self.f 
    
    def set_f(self, f : Callable[[int], int]) -> None:
        self.f = f
    
    def get_tolerance(self) -> Union[float, Fraction]:
        '''
        Returns tolerance of matrix. Basic getter.

        Parameters
        -------------
        None

        Returns
        -------------
        Union[float, Fraction]
            The tolerance of the matrix.
        '''
        return self.tolerance 
    
    def set_tolerance(self, tolerance : Union[float, Fraction]):
         '''
        Changes tolerance of matrix. Basic setter.

        Parameters
        -------------
        new_tolerance : Union[float, Fraction]
            New tolerance of matrix. 

        Returns
        -------------
        None
        '''
        self.tolerance = tolerance
