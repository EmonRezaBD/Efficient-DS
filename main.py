class BlockDiagonalMatrix:
    def __init__(self, n, d):
        self.n = n
        self.d = d
        # Initialize an n x n grid where each cell holds a list of d diagonal elements.
        self.blocks = [[None for _ in range(n)] for _ in range(n)]
    
    def set_block(self, i, j, diagonal_elements):
        """Sets the diagonal elements for block D_ij."""
        assert len(diagonal_elements) == self.d, "Diagonal must have exactly d elements"
        self.blocks[i][j] = diagonal_elements
    
    def get_block(self, i, j):
        """Returns the diagonal elements for block D_ij."""
        return self.blocks[i][j]
    
    def display(self):
        """Displays the full matrix representation of the block matrix."""
        # Compute the full matrix size
        matrix_size = self.n * self.d
        # Initialize a full matrix of zeros
        full_matrix = [[0] * matrix_size for _ in range(matrix_size)]
        
        # Fill in the full matrix with block diagonal values
        for i in range(self.n):
            for j in range(self.n):
                if self.blocks[i][j] is not None:
                    # Each block starts at (i*d, j*d) in the full matrix
                    for k in range(self.d):
                        full_matrix[i * self.d + k][j * self.d + k] = self.blocks[i][j][k]
        
        # Print the full matrix
        for row in full_matrix:
            print(" ".join(f"{val:5}" for val in row))


def multiply(A, B):
    assert A.n == B.n and A.d == B.d, "Matrices must have the same dimensions"
    n, d = A.n, A.d
    result = BlockDiagonalMatrix(n, d)
    
    # Perform block multiplication
    for i in range(n):
        for k in range(n):
            # Initialize an array to hold the diagonal elements of the result block
            result_block = [0] * d
            for j in range(n):
                # Multiply D_ij and D_jk if both are non-null
                if A.blocks[i][j] is not None and B.blocks[j][k] is not None:
                    for l in range(d):
                        result_block[l] += A.blocks[i][j][l] * B.blocks[j][k][l]
            result.set_block(i, k, result_block)
    
    return result

def invert_matrix(A):
    n, d = A.n, A.d
    inverse = BlockDiagonalMatrix(n, d)
    
    for i in range(n):
        for j in range(n):
            if i == j and A.blocks[i][j] is not None:
                inverse_block = [1 / x if x != 0 else 0 for x in A.blocks[i][j]]
                inverse.set_block(i, j, inverse_block)
            elif A.blocks[i][j] is not None:
                # If A has off-diagonal blocks, the matrix may not be block diagonal invertible.
                raise ValueError("Matrix is not invertible due to non-diagonal blocks.")
    
    return inverse

def main():
    # Initialize matrices A and B with dimensions and test blocks
    A = BlockDiagonalMatrix(n=2, d=3)
    B = BlockDiagonalMatrix(n=2, d=3)

    # Set diagonal elements for blocks
    A.set_block(0, 0, [1, 2, 3])
    A.set_block(1, 1, [4, 5, 6])
    B.set_block(0, 0, [7, 8, 9])
    B.set_block(1, 1, [10, 11, 12])

    # Multiply matrices
    C = multiply(A, B)
    # Invert matrix A if possible
    A_inverse = invert_matrix(A)
    C.display()
    print("\n\n")
    A_inverse.display()


if __name__=="__main__": #start of program
    main()

