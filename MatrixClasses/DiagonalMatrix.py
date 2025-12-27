from fractions import Fraction
from typing import Union
from collections.abc import Callable

class DiagonalMatrix:
    '''
    Class representing a diagonal matrix. 
    
    The diagonal entries are given via a callable N^2 -> C. 
    '''
    
    def __init__(self, diagonal : Callable[[int], Union[float, int, complex, Fraction]], tolerance : Union[float, Fraction]) -> None:
        '''
        Initialises the matrix A = (a_ij) by setting object attributes.
        
        Parameters
        -------------
        diagonal : Callable[[int], Union[float, int, complex, Fraction]]
            A callable giving the diagonal of A. 
        
        tolerance : Union[float, Fraction]
            Where a comparison with a float is required, this argument gives the offset used to account for floating point imprecision. 
            
            If abs(c) < tolerance, we will treat c as zero.
            
            Default argument is 1e-16 
        
        Raises
        -------------
        None 
        '''
        
        if not isinstance(tolerance, (float, Fraction)):
            raise TypeError("Tolerance must be float or Fraction")
        self.diagonal = diagonal 
        self.tolerance = tolerance 
    
    def __call__(self, i : int, j : int) -> Union[float, int, complex, Fraction]:
        '''
        Allows a DiagonalMatrix to be used a callable.
        
        If A = (a_ij) is a Diagonal Matrix, we now have A(i, j) = a_ij 
        
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
        return self.diagonal(i) if i == j else 0
    
    def __add__(self, B : 'DiagonalMatrix') -> 'DiagonalMatrix':
        '''
        Allows addition of two DiagonalMatrix objects by overloading the + operator. 

        In the following, we write A = (a_ij) = self for simplicity.

        Given a DiagonalMatrix B = (b_ij), we return A + B = (a_ij + b_ij).

        Parameters
        -------------
        B : DiagonalMatrix
            The B in A + B.

        Returns
        -------------
        DiagonalMatrix
            A + B as a DiagonalMatrix.

        Raises
        -------------
        TypeError
            If B is not a DiagonalMatrix. 
        '''   
        A = self 
        
        if not isinstance(B, DiagonalMatrix):
            raise TypeError("Cannot add DiagonalMatrix with object not of DiagonalMatrix type.")
               
        new_diagonal = lambda i : A.diagonal(i) + B.diagonal(i)
        new_tolerance = min(A.tolerance, B.tolerance)
        
        return DiagonalMatrix(new_diagonal, new_tolerance)
        
    def __rmul__(self, c : Union[float, int, complex, Fraction]) -> 'DiagonalMatrix':
        '''
        Allows left multiplication of DiagonalMatrix by a scalar by overloading the * operator.

        In the following, we write A = self for simplicity.

        Given a scalar multiple c, we return c*A as a DiagonalMatrix.
        
        Parameters
        -------------
        c : float, int, complex or Fraction
            Gives the scalar multiple.

        Returns
        -------------
        DiagonalMatrix
            c*A as a DiagonalMatrix.

        Raises
        -------------
        TypeError
            If c cannot be interpreted as a scalar.
        '''
        A = self 
        
        if not isinstance(c, (float, int, complex, Fraction)):
            raise TypeError("Cannot multiply SparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction")
        
        if abs(c) < A.tolerance: # c = 0
            # 0*A = 0
            return DiagonalMatrix(lambda i : 0, A.tolerance) 
        
        else:
            new_diagonal = lambda i : c*A.diagonal(i)
            return DiagonalMatrix(new_diagonal, A.tolerance)
        
    def __mul__(self, c : Union[float, int, complex, Fraction]) -> 'DiagonalMatrix':
        '''
        Allows right multiplication of DiagonalMatrix by a scalar by overloading the * operator.

        In the following, we write A = self for simplicity.

        Given a scalar multiple c, we return A*c = c*A as a DiagonalMatrix.
        
        Parameters
        -------------
        c : float, int, complex or Fraction
            Gives the scalar multiple.

        Returns
        -------------
        DiagonalMatrix
            A*c as a DiagonalMatrix.

        Raises
        -------------
        TypeError
            If c cannot be interpreted as a scalar.
        '''
        return self.__rmul__(c)
        
    def __matmul__(self, B : 'DiagonalMatrix') -> 'DiagonalMatrix':
        '''
        Allows multiplication of the DiagonalMatrix by a DiagonalMatrix by overloading the @ operator. 
        
        Parameters
        -------------
        B : DiagonalMatrix
            Gives B in A @ B.

        Returns
        -------------
        DiagonalMatrix
            Gives A @ B.

        Raises
        -------------
        TypeError
            If B is not a DiagonalMatrix.
        '''
        A = self 
        
        new_diagonal = lambda i : A.diagonal(i)*B.diagonal(i)
        new_tolerance = min(A.tolerance, B.tolerance)
        
        return DiagonalMatrix(new_diagonal, new_tolerance)
    
    def __sub__(self, B : 'DiagonalMatrix') -> 'DiagonalMatrix':
        '''
        Allows subtraction of two DiagonalMatrix objects by overloading the - operator. 

        In the following, we write A = (a_ij) = self for simplicity.

        Given a DiagonalMatrix B = (b_ij), we return A - B = (a_ij - b_ij).

        Parameters
        -------------
        B : DiagonalMatrix
            The B in A - B.

        Returns
        -------------
        DiagonalMatrix
            A - B as a DiagonalMatrix.

        Raises
        -------------
        TypeError
            If B is not a DiagonalMatrix. 
        '''   
        A = self 
        
        if not isinstance(B, DiagonalMatrix):
            raise TypeError("Cannot add DiagonalMatrix with object not of DiagonalMatrix type.")
               
        new_diagonal = lambda i : A.diagonal(i) - B.diagonal(i)
        new_tolerance = min(A.tolerance, B.tolerance)
        
        return DiagonalMatrix(new_diagonal, new_tolerance)
    
    def __repr__(self) -> str:
        '''
        Represent DiagonalMatrix as a string of the form:
            SparseMatrix(<string representation of self.diagonal>, self.tolerance)

        Parameters
        -------------
        None

        Returns
        -------------
        str
            String to print. 
        '''
        return f"SparseMatrix({self.diagonal}, {self.tolerance})"
    
    def PseudoInverse(self) -> DiagonalMatrix:
        '''
        Computes the Moore-Penrose pseudoinverse A^+ of the diagonal matrix A.
        
        This matrix satisfies the following properties:
            1. A @ A^+ @ A = A (maps column vectors of A to themselves)
            2. A^+ @ A @ A^+ = A (acts like a weak inverse) 
            3. (A @ A^+)^* = A @ A^+ (A @ A^+ is Hermitian)
            4. (A^+ @ A)^* = A^+ @ A (A^+ @ A is Hermitian)
        
        For a diagonal matrix, it is formed by taking the reciprocal of all non-zero matrix elements and fixing the zero elements.
        
        Parameters
        -------------
        None 
        
        
        Returns
        -------------
        DiagonalMatrix 
            Moore-Penrose pseudoinverse of A
        '''
        A = self 
        new_diagonal = lambda i : 0 if abs(A.diagonal(i)) < A.tolerance else 1/A.diagonal(i)
        return DiagonalMatrix(new_diagonal, A.tolerance) 
    
    def get_diagonal(self) -> Callable[[int], Union[float, int, complex, Fraction]]:
        '''
        Returns self.diagonal, the callable generating the diagonal of A = self. 
        
        Parameters
        -------------
        None 
        
        Returns 
        -------------
        Callable[[int], Union[float, int, complex, Fraction]]
            the callable giving the diagonal elements of A. 
        '''
        return self.diagonal 
    
    def set_diagonal(self, diagonal : Callable[[int], Union[float, int, complex, Fraction]]) -> None: 
        '''
        Sets self.diagonal, the callable generating the diagonal of A = self. 
        
        Parameters
        -------------
        Callable[[int], Union[float, int, complex, Fraction]]
            the callable giving the diagonal elements of A.  
        
        Returns 
        -------------
        None 
        '''
        self.diagonal = diagonal 
    
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
    
    def set_tolerance(self, tolerance : Union[float, Fraction]) -> None:
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
