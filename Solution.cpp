#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

class BlockDiagonalMatrix {
private:
    int n; // Number of blocks along one dimension
    int d; // Size of each block
    std::vector<std::vector<std::vector<int> > > blocks; // 3D vector to store diagonal elements

public:
    BlockDiagonalMatrix(int n, int d) : n(n), d(d) {
        blocks = std::vector<std::vector<std::vector<int> > >(n, std::vector<std::vector<int> >(n, std::vector<int>()));
    }

    void setBlock(int i, int j, const std::vector<int>& diagonalElements) {
        if (diagonalElements.size() != d) {
            throw std::invalid_argument("Diagonal must have exactly d elements");
        }
        blocks[i][j] = diagonalElements;
    }

    std::vector<int> getBlock(int i, int j) const {
        return blocks[i][j];
    }

    BlockDiagonalMatrix multiply(const BlockDiagonalMatrix& B) const {
        if (n != B.n || d != B.d) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }
        
        BlockDiagonalMatrix C(n, d);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!blocks[i][j].empty() && !B.blocks[i][j].empty()) {
                    std::vector<int> product(d);
                    for (int k = 0; k < d; ++k) {
                        product[k] = blocks[i][j][k] * B.blocks[i][j][k];
                    }
                    C.setBlock(i, j, product);
                }
            }
        }
        
        return C;
    }

    BlockDiagonalMatrix inverse() const {
        BlockDiagonalMatrix inv(n, d);
        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!blocks[i][j].empty()) {
                    std::vector<int> invDiagonal(d);
                    for (int k = 0; k < d; ++k) {
                        if (blocks[i][j][k] == 0) {
                            throw std::runtime_error("Matrix is singular and cannot be inverted");
                        }
                        invDiagonal[k] = 1 / blocks[i][j][k]; // Inverse each diagonal element
                    }
                    inv.setBlock(i, j, invDiagonal);
                }
            }
        }
        
        return inv;
    }

    void display() const {
        int matrixSize = n * d;
        std::vector<std::vector<int> > fullMatrix(matrixSize, std::vector<int>(matrixSize, 0));

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!blocks[i][j].empty()) {
                    for (int k = 0; k < d; ++k) {
                        fullMatrix[i * d + k][j * d + k] = blocks[i][j][k];
                    }
                }
            }
        }

        for (const auto &row : fullMatrix) {
            for (const auto &val : row) {
                std::cout << std::setw(5) << val;
            }
            std::cout << "\n";
        }
    }
};

int main() {
    int n = 2;
    int d = 3;

    BlockDiagonalMatrix A(n, d);
    A.setBlock(0, 0, std::vector<int>{1, 2, 3});
    A.setBlock(1, 1, std::vector<int>{4, 5, 6});

    BlockDiagonalMatrix B(n, d);
    B.setBlock(0, 0, std::vector<int>{7, 8, 9});
    B.setBlock(1, 1, std::vector<int>{10, 11, 12});

    std::cout << "Matrix A:\n";
    A.display();
    std::cout << "\nMatrix B:\n";
    B.display();

    std::cout << "\nA * B:\n";
    BlockDiagonalMatrix C = A.multiply(B);
    C.display();

    std::cout << "\nInverse of A:\n";
    try {
        BlockDiagonalMatrix A_inv = A.inverse();
        A_inv.display();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << "\n";
    }

    return 0;
}