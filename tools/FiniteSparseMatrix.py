class FiniteSparseMatrix:
    def __init__(self, entries : dict[tuple[int, int], float], default : float = 0.0, tolerance : float = 1e-16) -> None:
        self.entries = entries
        self.default = default 
        self.tolerance = tolerance
        
    def __call__(self, i : int, j : int) -> float:
        return self.entries.get((i, j), self.default)
    
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

    
