#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

#define BLOCK_SIZE 256 // Threads per block

// Kernel for matrix multiplication of diagonal blocks
__global__ void multiplyBlocksKernel(const double* A, const double* B, double* C, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d) {
        C[idx] = A[idx] * B[idx];
    }
}

// Kernel for matrix inversion of diagonal blocks
__global__ void invertBlocksKernel(const double* A, double* A_inv, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d) {
        if (A[idx] == 0) {
            printf("Matrix is singular at index %d, cannot invert!\n", idx);
            return;
        }
        A_inv[idx] = 1.0 / A[idx];
    }
}

class BlockDiagonalMatrix {
private:
    int n; // Number of blocks
    int d; // Size of each block
    std::vector<std::vector<std::vector<double>>> blocks;

public:
    // Constructor
    BlockDiagonalMatrix(int n, int d) : n(n), d(d) {
        blocks.resize(n, std::vector<std::vector<double>>(n, std::vector<double>(d, 0.0)));
    }

    // Set diagonal block
    void setBlock(int row, int col, const std::vector<double>& diagonalElements) {
        if (row >= n || col >= n) {
            throw std::out_of_range("Block index out of range");
        }
        if (diagonalElements.size() != d) {
            throw std::invalid_argument("Diagonal must have exactly d elements");
        }
        blocks[row][col] = diagonalElements;
    }

    // Get diagonal block
    std::vector<double> getBlock(int row, int col) const {
        if (row >= n || col >= n) {
            throw std::out_of_range("Block index out of range");
        }
        return blocks[row][col];
    }

    // Matrix multiplication
    BlockDiagonalMatrix multiply(const BlockDiagonalMatrix& B) const {
        if (n != B.n || d != B.d) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        BlockDiagonalMatrix C(n, d);

        // Flatten matrices for GPU processing
        std::vector<double> A_flat(n * d, 0);
        std::vector<double> B_flat(n * d, 0);
        std::vector<double> C_flat(n * d, 0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                A_flat[i * d + j] = blocks[i][i][j];
                B_flat[i * d + j] = B.blocks[i][i][j];
            }
        }

        // Allocate GPU memory
        double* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, n * d * sizeof(double));
        cudaMalloc(&d_B, n * d * sizeof(double));
        cudaMalloc(&d_C, n * d * sizeof(double));

        // Copy data to GPU
        cudaMemcpy(d_A, A_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int gridSize = (d + BLOCK_SIZE - 1) / BLOCK_SIZE; // Blocks needed per block diagonal
        multiplyBlocksKernel << <gridSize, BLOCK_SIZE >> > (d_A, d_B, d_C, d);

        // Copy result back to CPU
        cudaMemcpy(C_flat.data(), d_C, n * d * sizeof(double), cudaMemcpyDeviceToHost);

        // Reconstruct result matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                C.blocks[i][i][j] = C_flat[i * d + j];
            }
        }

        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return C;
    }

    // Matrix inversion
    BlockDiagonalMatrix inverse() const {
        BlockDiagonalMatrix inv(n, d);

        // Flatten matrices for GPU processing
        std::vector<double> A_flat(n * d, 0);
        std::vector<double> A_inv_flat(n * d, 0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                A_flat[i * d + j] = blocks[i][i][j];
            }
        }

        // Allocate GPU memory
        double* d_A, * d_A_inv;
        cudaMalloc(&d_A, n * d * sizeof(double));
        cudaMalloc(&d_A_inv, n * d * sizeof(double));

        // Copy data to GPU
        cudaMemcpy(d_A, A_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int gridSize = (d + BLOCK_SIZE - 1) / BLOCK_SIZE;
        invertBlocksKernel << <gridSize, BLOCK_SIZE >> > (d_A, d_A_inv, d);

        // Copy result back to CPU
        cudaMemcpy(A_inv_flat.data(), d_A_inv, n * d * sizeof(double), cudaMemcpyDeviceToHost);

        // Reconstruct result matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                inv.blocks[i][i][j] = A_inv_flat[i * d + j];
            }
        }

        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_A_inv);

        return inv;
    }

    // Display function
    /*void display() const {
        for (int i = 0; i < n; ++i) {
            for (double val : blocks[i][i]) {
                std::cout << val << " ";
            }
            std::cout << "\n";
        }
    }*/
    void display() const {
        for (int row = 0; row < n * d; ++row) {
            for (int col = 0; col < n * d; ++col) {
                int blockRow = row / d;
                int blockCol = col / d;
                int innerRow = row % d;
                int innerCol = col % d;

                if (blockRow == blockCol && innerRow == innerCol) {
                    std::cout << std::fixed << std::setprecision(4) << blocks[blockRow][blockCol][innerRow] << " ";
                }
                else {
                    std::cout << std::fixed << std::setprecision(4) << 0.0 << " ";
                }
            }
            std::cout << "\n";
        }
    }

};

int main() {
    int n = 2; // Number of diagonal blocks
    int d = 3; // Size of each block

    BlockDiagonalMatrix A(n, d);
    A.setBlock(0, 0, { 1.0, 2.0, 3.0 });
    A.setBlock(1, 1, { 4.0, 5.0, 6.0 });

    BlockDiagonalMatrix B(n, d);
    B.setBlock(0, 0, { 7.0, 8.0, 9.0 });
    B.setBlock(1, 1, { 10.0, 11.0, 12.0 });

    std::cout << "Matrix A:\n";
    A.display();

    std::cout << "\nMatrix B:\n";
    B.display();

    std::cout << "\nA * B:\n";
    BlockDiagonalMatrix C = A.multiply(B);
    C.display();

    std::cout << "\nInverse of A:\n";
    BlockDiagonalMatrix A_inv = A.inverse();
    A_inv.display();

    return 0;
}
