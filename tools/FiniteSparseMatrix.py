from fractions import Fraction
from typing import Union

class FiniteSparseMatrix:
    '''
    Class representing a sparse matrix with only finitely many non-zero entries. 

    Demonstrates how a matrix may be represented as a callable, with functions included to enable arithmetic. 
    '''
    def __init__(self, entries : dict[tuple[int, int], Union[float, int, complex, Fraction]], default : Union[float, int, complex, Fraction] = 0.0, tolerance : Union[float, Fraction] = 1e-16) -> None:
        '''
        Initialises the matrix by setting object attributes.

        Parameters
        -------------
        entries : dict[tuple[int, int], Union[float, int, complex, Fraction]]
            Dictionary giving the non-zero entries of the matrix. Can be given as floats, ints, complexes or Fractions or mixes of these. 
            
            Keys are the indices (i, j) of the non-zero elements, and the values are the corresponding elements.
        default : float, int, complex or Fraction
            Gives the "default value" for the unspecified elements of the matrix. Sparse matrices typically have these equal to zero. Can be float, int, complex or Fraction.
            
            Default value is 0.0
        tolerance : float or Fraction
            We recognize the x = 0 if x < tolerance to account for floating-point imprecision. Should be a very small number.

            Default value is 1e-16.

        Returns
        -------------
        None

        Raises
        -------------
        TypeError
            if default is not float, int, complex or Fraction
            if tolerance is not float or Fraction
        '''
        if not isinstance(default, (float, int, complex, Fraction)):
            raise TypeError("Default value must have type in [float, int, complex, Fraction]")
        if not isinstance(tolerance, (float, Fraction)):
            raise TypeError("Tolerance must be float or Fraction.")
        self.entries = entries
        self.default = default 
        self.tolerance = tolerance
        
    def __call__(self, i : int, j : int) -> Union[float, int, complex, Fraction]:
        '''
        Allows a FiniteSparseMatrix to be used as a callable. 

        If A is a FiniteSparseMatrix, then A(i, j) gives the (i, j)th element if it is a specified element in A.entries, else it will return the default value.

        Parameters
        -------------
        i : int
            Gives the row number of the requested element.
        j : int
            Gives the column number of the requested element.

        Returns
        -------------
        float
            The desired matrix element.

        Raises
        -------------
        None
        '''
        return self.entries.get((i, j), self.default) # tries self.entries[(i, j)]. if (i, j) is not a key in self.entries then we return self.default.
    
    def __add__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix': 
        # note that the string 'FiniteSparseMatrix' is used in the type hint as opposed to the type FiniteSparseMatrix.
        # Python checks whether the type in a type hint exists. 
        # Since we have not yet finished the definition of FiniteSparseMatrix, writing B : FiniteSparseMatrix and -> FiniteSparseMatrix would raise a NameError.
        '''
        Allows for the addition of two FiniteSparseMatrixs by overloading the + operator. 

        In the following, we write A = self for simplicity.

        Given a FiniteSparseMatrix B, we return A + B.

        Parameters
        -------------
        B : FiniteSparseMatrix
            The B in A + B.

        Returns
        -------------
        FiniteSparseMatrix
            A + B as a FiniteSparseMatrix.

        Raises
        -------------
        TypeError
            If B is not a FiniteSparseMatrix. 
        '''
        if not isinstance(B, FiniteSparseMatrix):
            raise TypeError("Cannot add FiniteSparseMatrix with object not of FiniteSparseMatrix type.")
        A = self # for the sake of clarity. Note that A is another name for self, and does not copy the object.
        new_default = A.default + B.default # for (i, j) not specified in either A or B, the value will simply be the sum of the defaults for A and B.
        A_size = len(A.entries) 
        B_size = len(B.entries)
        # we use A_size and B_size to determine which of A and B have specified elements. We will start by copying the larger matrix, and then add the entries of the smaller matrix
        # this may be significantly cheaper if one matrix has a large number of specified elements and the other has few specified elements.
        smaller, bigger = A, B if A_size <= B_size else B, A
        new_entries = bigger.entries.copy() 
        new_tolerance = min(A.tolerance, B.tolerance)
        
        for idx_i, idx_j in smaller.entries:
            candidate = bigger(idx_i, idx_j) + smaller(idx_i, idx_j)
            idx = (idx_i, idx_j)
            if abs(candidate - new_default) < new_tolerance: 
                # if the sum is equal to the new default up to the specified tolerance, we treat it as a default value and hence pop it from the dictionary of specified values
                new_entries.pop(idx, None)
                
            else:
                # else, we write it as a new specified value
                new_entries[idx] = candidate 
                
        return FiniteSparseMatrix(new_entries, new_default, new_tolerance)

    def __mul__(self, c : Union[float, int, complex, Fraction]) -> 'FiniteSparseMatrix':
        '''
        Allows for the multiplication of FiniteSparseMatrix by a scalar by overloading the * operator.

        In the following, we write A = self for simplicity.

        Given a scalar multiple c, we return c*A as a FiniteSparseMatrix.
        
        Parameters
        -------------
        c : float, int, complex or Fraction
            Gives the scalar multiple.

        Returns
        -------------
        FiniteSparseMatrix
            c*A as a FiniteSparseMatrix.

        Raises
        -------------
        TypeError
            If c cannot be interpreted as a scalar.
        '''
        A = self # for the sake of clarity. Note that A is another name for self, and does not copy the object.
        if not isinstance(c, (float, int, complex, Fraction)):
            raise TypeError("Cannot multiply FiniteSparseMatrix with non-scalar. Acceptable scalar types are float, int, complex, fractions.Fraction")
            
        if abs(c) < self.tolerance:
            # if abs(c) < tolerance, then we treat c = 0 and hence output the zero matrix
            return FiniteSparseMatrix({}, 0.0)
            
        else:
            # else, we multiply all elements of the matrix (including the defaults) by c. 
            new_entries = {key : c*val for key, val in A.entries.items()}
            new_default = c*A.default
            return FiniteSparseMatrix(new_entries, new_default, self.tolerance)

    def __matmul__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        '''
        Allows for the multiplication of FiniteSparseMatrix by a FiniteSparseMatrix by overloading the @ operator. 
        
        This is mathematically guaranteed to produce another FiniteSparseMatrix if the default values of both matrices are zero.

        In the following, we write A = self for simplicity.

        Given a FiniteSparseMatrix B, we find the matrix product A @ B.

        Parameters
        -------------
        B : FiniteSparseMatrix
            Gives B in A @ B.

        Returns
        -------------
        FiniteSparseMatrix
            A @ B as a FiniteSparseMatrix.

        Raises
        -------------
        TypeError
            If B is not a FiniteSparseMatrix.
        ValueError
            If the default value of either A or B is non-zero. 
        '''
        A = self  # for the sake of clarity. Note that A is another name for self, and does not copy the object.
        if not isinstance(B, FiniteSparseMatrix):
            raise TypeError("Cannot multiply FiniteSparseMatrix with object not of FiniteSparseMatrix type.")
        
        if B.default != 0.0 or A.default != 0.0:
            raise ValueError("Default value of both self and B should be 0.0")
            
        if not B.entries or not A.entries: 
            # if A = 0 or B = 0, product is zero so we can return zero matrix with no calculation.
            return FiniteSparseMatrix({}, 0.0, A.tolerance)

        # note that if A @ B = (c_ij), we have c_ij = Σ_k a_ik b_kj. Initially, this is an infinite sum and infinitely many (i, j) are concerned.
        # [1] Note that c_ij can only be non-zero if both the ith row of A and the jth column of B are non-zero. 
        # [2] furthermore, we only need to run over the non-zero columns of A and the non-zero rows of B, restricting the k that we need to sum over.
        
        A_non_zero_rows = set(elt[0] for elt, _ in A.entries.items()) # create a set of the indices of the non-zero rows of A
        
        B_non_zero_rows = set(elt[0] for elt, _ in B.entries.items()) # create a set of the indices of the non-zero rows of B
        B_non_zero_rows_max = max(B_non_zero_rows) # find the maximum index of a non-zero row of B
        
        A_non_zero_cols = set(elt[1] for elt, _ in A.entries.items()) # create a set of the indices of the non-zero columns of A
        A_non_zero_cols_max = max(A_non_zero_cols) # find the maximum index of a non-zero colum of A
        
        B_non_zero_cols = set(elt[1] for elt, _ in B.entries.items()) # create a set of the indices of the non-zero columns of B
        
        upper_bound = min(A_non_zero_cols_max, B_non_zero_rows_max) + 1 # by comment [2], we only need to consider k up to this minimum.

        new_entries = {}

        new_tolerance = min(A.tolerance, B.tolerance)
        
        for i in A_non_zero_rows:
            for j in B_non_zero_cols:
                # c_ij can only be non-zero if i is in A_non_zero_rows and j is in B_non_zero_cols by comment [1]. 
                candidate = 0.0
                for k in range(0, upper_bound):
                    candidate += A(i, k)*B(k, j)
                    
                if abs(candidate) > new_tolerance:
                    # if the computed matrix element is distinguished from zero, we note it as a distinguished value.
                    new_entries[(i, j)] = candidate

        return FiniteSparseMatrix(new_entries, 0.0, new_tolerance)

    def __sub__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        A = self
        return A + (-1)*B

    def __repr__(self) -> str:
        return f"FiniteSparseMatrix({len(self.entries)} entries, {self.default})"
       
    def get_entries(self) -> dict[tuple[int, int], Union[float, int, complex, Fraction]]:
        return self.entries 
    
    def set_entries(self, new_entries : dict[tuple[int, int], Union[float, int, complex, Fraction]]) -> None:
        for idx, val in new_entries.items():
            self.entries[idx] = val

    def pop(self, pos : dict[tuple[int, int], Union[float, int, complex, Fraction]]) -> Union[None, float, int, complex, Fraction]:
        return self.entries.pop(pos, None)
    
    def get_default(self):
        return self.default
    
    def set_default(self, new_default : float) -> None:
        self.default = new_default

    def get_tolerance(self):
        return self.tolerance

    def set_tolerance(self, new_tolerance : Union[float, Fraction]) -> None:
        self.tolerance = new_tolerance
    
