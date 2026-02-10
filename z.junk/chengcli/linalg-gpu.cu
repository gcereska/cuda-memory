#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <assert.h>

#include <memorypool/alloc.h>
#include <memorypool/math/linalg.h>

// test vvdot
// TODO: could change errors to exit(1) for better ctest
// test ludcmp
__device__ void test_ludcmp()
{
  // printf("Testing ludcmp...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  int d = ludcmp(a, indx, n);
  
  assert(d == -1);

  // printf("LU decomposition: d = %d\n", d);
  // printf("LU matrix:\n");
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%f ", a[i * n + j]);
  //   }
  //   printf("\n");
  // }
  // printf("Permutation vector: ");
  // for (int i = 0; i < n; i++) {
  //   printf("%d ", indx[i]);
  // }
  // printf("\n");
  // printf("\n");
}

// test lubksb
__device__ void test_lubksb()
{
  // printf("Testing lubksb...\n");
  double b[2] = {5.0, 11.0};
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;

  // printf("matrix A= \n");
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%f ", a[i * n + j]);
  //   }
  //   printf("\n");
  // }

  // printf("vector b= \n");
  // for (int i = 0; i < n; i++) {
  //   printf("%f\n", b[i]);
  // }

  ludcmp(a, indx, n);
  lubksb(b, a, indx, n);
  // printf("Solution vector x= \n");
  // for (int i = 0; i < n; i++) {
  //   printf("%f\n", b[i]);
  // }
  // printf("\n");
  
  assert(b[0] == 1.0);
  assert(b[1] == 2.0);
}

// test luminv
__device__ void test_luminv()
{
  // printf("Testing luminv...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  double y[4];

  // printf("matrix A= \n");
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%f ", a[i * n + j]);
  //   }
  //   printf("\n");
  // }

  ludcmp(a, indx, n);
  luminv(y, a, indx, n);
  // printf("Inverse matrix Y= \n");
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     printf("%f ", y[i * n + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  double expected[4] = { -2.0, 1.0, 1.5, -0.5 };
  for (int i = 0; i < n * n; i++) {
    assert(fabs(y[i] - expected[i]) < 1e-6);
  }

}

// test leastsq
__device__ void test_leastsq()
{
  // printf("Testing leastsq...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;

  // printf("Matrix A= \n");
  // for (int i = 0; i < n1; i++) {
  //   for (int j = 0; j < n2; j++) {
  //     printf("%f ", a[i * n2 + j]);
  //   }
  //   printf("\n");
  // }

  // printf("Vector B= \n");
  // for (int i = 0; i < n1; i++) {
  //   printf("%f\n", b[i]);
  // }

  leastsq(b, a, n1, n2);
  // printf("Least squares solution: \n");
  // for (int i = 0; i < n2; i++) {
  //   printf("%f\n", b[i]);
  // }
  // printf("\n");
  
  double expected[2] = { -6.0, 6.5 };
  for (int i = 0; i < n2; i++) {
    assert(fabs(b[i] - expected[i]) < 1e-6);
  }

}

// test leastsq_kkt
__device__ void test_leastsq_kkt()
{
  // printf("Testing leastsq_kkt...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double c[4] = {7.0, 8.0, 9.0, 10.0};
  double d[2] = {11., 15.};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;
  int n3 = 2;
  int neq = 1;
  
  // printf("Matrix A= \n");
  // for (int i = 0; i < n1; i++) {
  //   for (int j = 0; j < n2; j++) {
  //     printf("%f ", a[i * n2 + j]);
  //   }
  //   printf("\n");
  // }

  // printf("Vector B= \n");
  // for (int i = 0; i < n1; i++) {
  //   printf("%f\n", b[i]);
  // }

  // printf("Matrix C= \n");
  // for (int i = 0; i < n3; i++) {
  //   for (int j = 0; j < n2; j++) {
  //     printf("%f ", c[i * n2 + j]);
  //   }
  //   printf("\n");
  // }
  // printf("neq = %d\n", neq);

  // printf("Vector D= \n");
  // for (int i = 0; i < n3; i++) {
  //   printf("%f\n", d[i]);
  // }

  int max_iter = 20;
  int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  if (err != 0) {
    printf("Error in leastsq_kkt: %d\n", err);
  }
  
  // printf("Constrained least squares solution: \n");
  // for (int i = 0; i < n2; i++) {
  //   printf("%f\n", b[i]);
  // }
  // printf("Number of iterations: %d\n", max_iter);
  // printf("\n");
  
  double expected[2] = { -5.285714, 6.0 };
  for (int i = 0; i < n2; i++) {
    assert(fabs(b[i] - expected[i]) < 0.0001);
  }
  assert(max_iter == 1);
}

__global__ void test_linalg_kernel(void *ptr) {
    #ifdef USE_MEMORY_POOL
    unsigned int threadInd = threadIdx.x + blockIdx.x * blockDim.x;
    poolinit(ptr, threadInd);
    #endif
    
    test_ludcmp();
    test_lubksb();
    test_luminv();
    test_leastsq();
    test_leastsq_kkt();
}

void run_linalg_tests() {
    #ifdef USE_MEMORY_POOL
    void *ptr = allocatePools(256);
    test_linalg_kernel<<<1, 256>>>(ptr);
    #else
    test_linalg_kernel<<<1, 256>>>(NULL);
    #endif

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel execution: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    run_linalg_tests();
    cudaDeviceSynchronize();
    printf("Done!\n");
}

