#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

class BlockDiagonalMatrix {
private:
    int n; // Number of blocks along one dimension
    int d; // Size of each block
    std::vector<std::vector<std::vector<double> > > blocks; // Changed int to double

public:
    BlockDiagonalMatrix(int n, int d) : n(n), d(d) {
        blocks = std::vector<std::vector<std::vector<double> > >(n, std::vector<std::vector<double> >(n, std::vector<double>()));
    }

    void setBlock(int i, int j, const std::vector<double>& diagonalElements) {
        if (diagonalElements.size() != d) {
            throw std::invalid_argument("Diagonal must have exactly d elements");
        }
        blocks[i][j] = diagonalElements;
    }

    std::vector<double> getBlock(int i, int j) const {
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
                    std::vector<double> product(d);
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
                    std::vector<double> invDiagonal(d);
                    for (int k = 0; k < d; ++k) {
                        if (blocks[i][j][k] == 0) {
                            throw std::runtime_error("Matrix is singular and cannot be inverted");
                        }
                        invDiagonal[k] = 1.0 / blocks[i][j][k]; // Changed to 1.0 to ensure floating-point division
                    }
                    inv.setBlock(i, j, invDiagonal);
                }
            }
        }
        
        return inv;
    }

    void display() const {
        int matrixSize = n * d;
        std::vector<std::vector<double> > fullMatrix(matrixSize, std::vector<double>(matrixSize, 0.0)); // Changed int to double

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
                std::cout << std::setw(10) << std::fixed << std::setprecision(4) << val;
            }
            std::cout << "\n";
        }
    }
};

int main() {
    int n = 2;
    int d = 3;

    BlockDiagonalMatrix A(n, d);
    A.setBlock(0, 0, std::vector<double>{1, 2, 3});
    A.setBlock(1, 1, std::vector<double>{4, 5, 6});

    BlockDiagonalMatrix B(n, d);
    B.setBlock(0, 0, std::vector<double>{7, 8, 9});
    B.setBlock(1, 1, std::vector<double>{10, 11, 12});

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
