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
        if not isinstance(tolerance, (float, Fraction)):
            raise TypeError("Tolerance must be float or Fraction")
        self.entries = entries 
        self.f = f
        self.tolerance = tolerance 
    
    def __call__(self, i : int, j : int) -> Union[float, int, complex, Fraction]:
        return 0 if (j > self.f(i) or i > self.f(j)) else self.entries(i, j)
    
    def __add__(self, B : 'SparseMatrix') -> 'SparseMatrix':
        A = self 
        
        if not isinstance(B, FiniteSparseMatrix):
            raise TypeError("Cannot add SparseMatrix with object not of SparseMatrix type.")
        
        new_entries = lambda i, j : A.entries(i, j) + B.entries(i, j)
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
    
    def get_tolerance(self) -> Union[float, Fraction]:
        return self.tolerance 
    
    def set_tolerance(self, tolerance : Union[float, Fraction]):
        self.tolerance = tolerance
