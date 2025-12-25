class SparseMatrix:
    def __init__(self, non_zero_entries : dict[tuple[int, int], float], default_value : float = 0.0):
        self.entries = non_zero_entries
        self.default = default_value 
        
    def __call__(self, i : int, j : int) -> float:
        return self.entries.get((i, j), self.default)
    
    def __add__(self, B : 'SparseMatrix') -> 'SparseMatrix':
        # TODO: identify smaller matrix between self and B, and for loop over that instead of B by default.
        new_default = self.default + B.default
        new_entries = self.entries.copy()
        for idx in B.entries:
            candidate = new_entries.get(idx, 0.0) + B.entries[idx]
            if abs(candidate - new_default) < 1e-16:
                new_entries.pop(idx, None)
            else:
                new_entries[idx] = candidate                
        return SparseMatrix(new_entries, new_default)

    def __repr__(self) -> str:
        return f"SparseMatrix({self.entries} (len {len(self.entries)}), {self.default})"
       
    def get_entries(self) -> dict[tuple[int, int], float]:
        return self.entries 
    
    def set_entries(self, new_entries : dict[tuple[int, int], float]) -> None:
        self.entries = new_entries
    
    def get_default(self):
        return self.default
    
    def set_default(self, new_default : float) -> None:
        self.default = new_default
    
    def cleanup_defaults(self, tolerance : float) -> None:
        pass
