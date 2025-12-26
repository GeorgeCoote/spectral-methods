from typing import Union

class FiniteSparseMatrix:
    '''
    Class representing a sparse matrix with only finitely many non-zero entries. 

    Demonstrates how a matrix may be represented as a callable, with functions included to enable arithmetic. 
    '''
    def __init__(self, entries : dict[tuple[int, int], float], default : Union[float, int] = 0.0, tolerance : float = 1e-16) -> None:
        '''
        Initialises the matrix by setting object attributes.

        Parameters
        -------------
        entries : dict[tuple[int, int], float]
            Dictionary giving the non-zero entries of the matrix. Keys are the indices (i, j) of the non-zero elements, and the values are the corresponding elements.
        default : Union[float, int]
            Gives the "default value" for the unspecified elements of the matrix. Sparse matrices typically have these equal to zero. Can be float or int.
            
            Default value is 0.0
        tolerance : float
            We recognize the x = 0 if x < tolerance to account for floating-point imprecision. Should be a very small number.

            Default value is 1e-16.

        Returns
        -------------
        None

        Raises
        -------------
        TypeError
            if default is not float or int
            if tolerance is not float
        '''
        if not isinstance(default, (float, int)):
            raise TypeError("Default value must be float or int.")
        if not isinstance(tolerance, float):
            raise TypeError("Tolerance must be float.")
        self.entries = entries
        self.default = default 
        self.tolerance = tolerance
        
    def __call__(self, i : int, j : int) -> float:
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
        new_default = self.default + B.default
        self_size = len(self.entries)
        B_size = len(B.entries)
        smaller, bigger = self, B if self_size <= B_size else B, self
        new_entries = bigger.entries.copy()
        
        for idx in smaller.entries:
            candidate = new_entries.get(idx, 0.0) + smaller.entries[idx]
            if abs(candidate - new_default) < self.tolerance:
                new_entries.pop(idx, None)
                
            else:
                new_entries[idx] = candidate 
                
        return FiniteSparseMatrix(new_entries, new_default)

    def __mul__(self, c : float) -> 'FiniteSparseMatrix':
        if c < self.tolerance:
            return FiniteSparseMatrix({}, 0.0)
            
        else:
            new_entries = {key : c*val for key, val in self.entries.items()}
            new_default = c*self.default
            return FiniteSparseMatrix(new_entries, new_default)

    def __matmul__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        if B.default != 0.0 or self.default != 0.0:
            raise ValueError("Default value of both self and B should be 0.0")
            
        if not B.entries or not self.entries:
            return FiniteSparseMatrix({}, 0.0)
            
        i_vals_self = set([elt[0] for elt, _ in self.entries.items()])
        
        i_vals_B = set([elt[0] for elt, _ in B.entries.items()])
        max_i_vals_B = max(i_vals_B)
        
        j_vals_self = set([elt[1] for elt, _ in self.entries.items()])
        max_j_vals_self = max(j_vals_self)
        
        j_vals_B = set([elt[1] for elt, _ in B.entries.items()])
        
        upper_bound = min(max_i_vals_B, max_j_vals_self) + 1

        new_entries = {}
        
        for i in i_vals_self:
            for j in j_vals_B:
                candidate = 0.0
                for k in range(0, upper_bound):
                    candidate += self(i, k)*B(k, j)
                    
                if abs(candidate) > self.tolerance:
                    new_entries[(i, j)] = candidate

        return FiniteSparseMatrix(new_entries, 0.0)

    def __sub__(self, B : 'FiniteSparseMatrix') -> 'FiniteSparseMatrix':
        return A + (-1)*B

    def __repr__(self) -> str:
        return f"FiniteSparseMatrix({len(self.entries)} entries, {self.default})"
       
    def get_entries(self) -> dict[tuple[int, int], float]:
        return self.entries 
    
    def set_entries(self, new_entries : dict[tuple[int, int], float]) -> None:
        for idx, val in new_entries.items():
            self.entries[idx] = val

    def pop(self, pos : dict[tuple[int, int], float]) -> None:
        self.entries.pop(pos, None)
    
    def get_default(self):
        return self.default
    
    def set_default(self, new_default : float) -> None:
        self.default = new_default

    
