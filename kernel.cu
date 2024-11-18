#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <iomanip>

// Kernel for block matrix multiplication
__global__ void blockMatrixMultKernel(const double* A, const double* B, double* C,
    int n, int d, int block_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = threadIdx.z;

    if (row < n && col < n && vec_idx < d) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            // Calculate indices for block matrices
            int a_idx = (row * n + k) * d + vec_idx;
            int b_idx = (k * n + col) * d + vec_idx;
            sum += A[a_idx] * B[b_idx];
        }
        int c_idx = (row * n + col) * d + vec_idx;
        C[c_idx] = sum;
    }
}

__global__ void elementWiseInverseKernel(const double* input, double* output, 
                                       int n, int d /*,bool* error_flag*/) {
    int block_i = blockIdx.x;  // block row index
    int block_j = blockIdx.y;  // block column index
    int element_idx = threadIdx.x;  // element index within block

    if (block_i < n && block_j < n && element_idx < d) {
        int idx = (block_i * n + block_j) * d + element_idx;
        double value = input[idx];
       /* 
        if (value == 0.0) {
            *error_flag = true;
            return;
        }
        */
        output[idx] = 1.0 / value;
    }
}

class BlockDiagonalMatrix {
private:
    int n, d;
    std::vector<std::vector<std::vector<double>>> blocks;

public:
    BlockDiagonalMatrix(int n_, int d_) : n(n_), d(d_) {
        blocks.resize(n, std::vector<std::vector<double>>(n, std::vector<double>(d)));
    }

    void setBlock(int i, int j, const std::vector<double>& values) {
        if (i < n && j < n && values.size() == d) {
            blocks[i][j] = values;
        }
    }

    std::vector<double> flatten() const {
        std::vector<double> flat(n * n * d);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < d; k++) {
                    flat[(i * n + j) * d + k] = blocks[i][j][k];
                }
            }
        }
        return flat;
    }

    void fromFlattened(const std::vector<double>& flat) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < d; k++) {
                    blocks[i][j][k] = flat[(i * n + j) * d + k];
                }
            }
        }
    }

  /*  void print() const {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << "Block[" << i << "][" << j << "]: ";
                for (double val : blocks[i][j]) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }
    }*/

    void printFormatted() const {
        // Create a 6x6 matrix (for n=2, d=3)
        std::vector<std::vector<double>> fullMatrix(n * d, std::vector<double>(n * d, 0.0));

        // Fill the matrix with block values
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < d; k++) {
                    // Each block value goes into its corresponding position
                    fullMatrix[i * d + k][j * d + k] = blocks[i][j][k];
                }
            }
        }

        // Print the full matrix
        for (int i = 0; i < n * d; i++) {
            std::cout << " ";
            for (int j = 0; j < n * d; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << fullMatrix[i][j];
            }
            std::cout << std::endl;
        }
    }
};

// Function to multiply block matrices using CUDA
void multiplyBlockMatrices(const BlockDiagonalMatrix& A, const BlockDiagonalMatrix& B,
    BlockDiagonalMatrix& C, int n, int d) {
    // Flatten matrices for CUDA processing
    std::vector<double> a_flat = A.flatten();
    std::vector<double> b_flat = B.flatten();
    std::vector<double> c_flat(n * n * d);

    // Allocate device memory
    double* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, n * n * d * sizeof(double));
    cudaMalloc(&d_B, n * n * d * sizeof(double));
    cudaMalloc(&d_C, n * n * d * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, a_flat.data(), n * n * d * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b_flat.data(), n * n * d * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(8, 8, d);  // Adjust these values based on your GPU
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
        (n + blockDim.y - 1) / blockDim.y,
        1);

    //For inverse
    double *d_output;
    cudaMalloc(&d_output, n * n * d * sizeof(double));
    std::vector<double> flat_output(n * n * d);


    // Launch kernel
    blockMatrixMultKernel << <gridDim, blockDim >> > (d_A, d_B, d_C, n, d, d);
    elementWiseInverseKernel << <gridDim, blockDim >> > (d_A, d_output, n, d);

    cudaMemcpy(flat_output.data(), d_output, n * n * d * sizeof(double),
        cudaMemcpyDeviceToHost);

    // Copy result back to host
    cudaMemcpy(c_flat.data(), d_C, n * n * d * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Convert back to block matrix format
    C.fromFlattened(c_flat);
}

int main() {
    int n = 2;  // 2x2 block matrix
    int d = 3;  // Each block is a vector of size 3

    // Initialize matrices A and B
    BlockDiagonalMatrix A(n, d);
    A.setBlock(0, 0, std::vector<double>{1, 2, 3});
   // A.setBlock(0, 1, std::vector<double>{4, 5, 6});
    //A.setBlock(1, 0, std::vector<double>{40, 50, 60});
    A.setBlock(1, 1, std::vector<double>{10, 20, 30});

    BlockDiagonalMatrix B(n, d);
    B.setBlock(0, 0, std::vector<double>{7, 8, 9});
  //  B.setBlock(0, 1, std::vector<double>{10, 11, 12});
   // B.setBlock(1, 0, std::vector<double>{-4, -5, -6});
    B.setBlock(1, 1, std::vector<double>{2, 3, 4});

    // Create result matrix
    BlockDiagonalMatrix C(n, d);

    // Perform multiplication
    multiplyBlockMatrices(A, B, C, n, d);

    // Print results
    std::cout << "Matrix A:" << std::endl;
    A.printFormatted();

    std::cout << "\nMatrix B:" << std::endl;
    B.printFormatted();

    std::cout << "\nA * B:" << std::endl;
    C.printFormatted();

    std::cout << "\nA inserve:" << std::endl;
    A.printFormatted();


    return 0;
}
