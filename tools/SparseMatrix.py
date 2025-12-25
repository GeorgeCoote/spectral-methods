class SparseMatrix:
    def __init__(self, non_zero_entries : dict[tuple[int, int], float], default_value : float = 0.0):
        self.entries = non_zero_entries
        self.default = default_value 
        
    def __call__(self, i : int, j : int) -> float:
        return self.entries.get((i, j), self.default)
    
    def __add__(self, B : SparseMatrix) -> SparseMatrix:
        new_entries = self.entries.copy()
        for idx in B.entries:
            new_entries[idx] = new_entries.get(idx, 0.0) + B.entries[idx] 
        new_default = self.default + B.get_default
        return SparseMatrix(new_entries, new_default)
       
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
