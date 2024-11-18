
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>

#define BLOCK_SIZE 256 // Adjust this for optimization based on your GPU

__global__ void multiplyBlocksKernel(double* A, double* B, double* C, int d, int numBlocks) {
    int blockId = blockIdx.x; // Each block handles one diagonal block
    int threadId = threadIdx.x;

    if (blockId < numBlocks) {
        int offset = blockId * d; // Start index of this block's diagonal
        for (int i = threadId; i < d; i += blockDim.x) {
            C[offset + i] = A[offset + i] * B[offset + i];
        }
    }
}

__global__ void invertBlocksKernel(double* A, double* A_inv, int d, int numBlocks) {
    int blockId = blockIdx.x; // Each block handles one diagonal block
    int threadId = threadIdx.x;

    if (blockId < numBlocks) {
        int offset = blockId * d; // Start index of this block's diagonal
        for (int i = threadId; i < d; i += blockDim.x) {
            if (A[offset + i] == 0) {
                printf("Matrix is singular, cannot invert!\n");
                return;
            }
            A_inv[offset + i] = 1.0 / A[offset + i];
        }
    }
}

class BlockDiagonalMatrix {
private:
    int n;  // Number of blocks
    int d;  // Size of each block
    std::vector<std::vector<double>> blocks; // Diagonal blocks

public:
    BlockDiagonalMatrix(int n, int d) : n(n), d(d) {
        blocks.resize(n, std::vector<double>(d, 0.0));
    }

    void setBlock(int blockIndex, const std::vector<double>& diagonalElements) {
        if (diagonalElements.size() != d) {
            throw std::invalid_argument("Diagonal must have exactly d elements");
        }
        blocks[blockIndex] = diagonalElements;
    }

    std::vector<double> getBlock(int blockIndex) const {
        return blocks[blockIndex];
    }

    BlockDiagonalMatrix multiply(const BlockDiagonalMatrix& B) const {
        if (n != B.n || d != B.d) {
            throw std::invalid_argument("Matrix dimensions must match for multiplication");
        }

        BlockDiagonalMatrix C(n, d);

        // Flatten matrices for GPU
        std::vector<double> A_flat(n * d);
        std::vector<double> B_flat(n * d);
        std::vector<double> C_flat(n * d);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                A_flat[i * d + j] = blocks[i][j];
                B_flat[i * d + j] = B.blocks[i][j];
            }
        }

       //  Allocate GPU memory
        double* d_A, * d_B, * d_C;
        cudaMalloc(&d_A, n * d * sizeof(double));
        cudaMalloc(&d_B, n * d * sizeof(double));
        cudaMalloc(&d_C, n * d * sizeof(double));

        // Copy data to GPU
        cudaMemcpy(d_A, A_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int numBlocks = n;
        multiplyBlocksKernel << <numBlocks, BLOCK_SIZE >> > (d_A, d_B, d_C, d, n);

        // Copy result back to CPU
        cudaMemcpy(C_flat.data(), d_C, n * d * sizeof(double), cudaMemcpyDeviceToHost);

        // Reconstruct result matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                C.blocks[i][j] = C_flat[i * d + j];
            }
        }

        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        return C;
    }

    BlockDiagonalMatrix inverse() const {
        BlockDiagonalMatrix inv(n, d);

        // Flatten matrices for GPU
        std::vector<double> A_flat(n * d);
        std::vector<double> A_inv_flat(n * d);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                A_flat[i * d + j] = blocks[i][j];
            }
        }

        // Allocate GPU memory
        double* d_A, * d_A_inv;
        cudaMalloc(&d_A, n * d * sizeof(double));
        cudaMalloc(&d_A_inv, n * d * sizeof(double));

       //  Copy data to GPU
        cudaMemcpy(d_A, A_flat.data(), n * d * sizeof(double), cudaMemcpyHostToDevice);

       //  Launch kernel
        int numBlocks = n;
        invertBlocksKernel << <numBlocks, BLOCK_SIZE >> > (d_A, d_A_inv, d, n);

       //  Copy result back to CPU
        cudaMemcpy(A_inv_flat.data(), d_A_inv, n * d * sizeof(double), cudaMemcpyDeviceToHost);

        // Reconstruct result matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                inv.blocks[i][j] = A_inv_flat[i * d + j];
            }
        }

        // Free GPU memory
        cudaFree(d_A);
        cudaFree(d_A_inv);

        return inv;
    }

    void display() const {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < d; ++j) {
                std::cout << blocks[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
};

int main() {
    int n = 2; // Number of blocks
    int d = 3; // Size of each block

    BlockDiagonalMatrix A(n, d);
    A.setBlock(0, { 1.0, 2.0, 3.0 });
    A.setBlock(1, { 4.0, 5.0, 6.0 });

    BlockDiagonalMatrix B(n, d);
    B.setBlock(0, { 7.0, 8.0, 9.0 });
    B.setBlock(1, { 10.0, 11.0, 12.0 });

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


