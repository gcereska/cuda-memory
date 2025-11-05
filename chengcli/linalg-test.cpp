#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memorypool/alloc.h>
#include <memorypool/math/linalg.h>

// test vvdot
// TODO: could change errors to exit(1) for better ctest
void test_vvdot()
{
  printf("Testing vvdot...\n");
  double a[3] = {1.0, 2.0, 3.0};
  double b[3] = {4.0, 5.0, 6.0};
  double result = vvdot(a, b, 3);
  printf("Dot product: %f\n", result);
  printf("Expected: 32.000000\n");
  printf("\n");
}

// test mvdot
void test_mvdot()
{
  printf("Testing mvdot...\n");

  double m[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  printf("Matrix = \n");
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%f ", m[i * 3 + j]);
    }
    printf("\n");
  }

  double v[3] = {7.0, 8.0, 9.0};
  printf("Vector = \n");
  for (int i = 0; i < 3; i++) {
    printf("%f\n", v[i]);
  }

  double r[2];
  mvdot(r, m, v, 2, 3);
  printf("Matrix-vector product: %f %f\n", r[0], r[1]);
  printf("\n");
}

// test mmdot
void test_mmdot()
{
  printf("Testing mmdot...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double r[4];
  int n1 = 2;
  int n2 = 3;
  int n3 = 2;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Matrix B= \n");
  for (int i = 0; i < n2; i++) {
    for (int j = 0; j < n3; j++) {
      printf("%f ", b[i * n3 + j]);
    }
    printf("\n");
  }

  mmdot(r, a, b, n1, n2, n3);
  
  printf("Matrix product R= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n3; j++) {
      printf("%f ", r[i * n3 + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// test ludcmp
void test_ludcmp()
{
  printf("Testing ludcmp...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  int d = ludcmp(a, indx, n);
  printf("LU decomposition: d = %d\n", d);
  printf("LU matrix:\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }
  printf("Permutation vector: ");
  for (int i = 0; i < n; i++) {
    printf("%d ", indx[i]);
  }
  printf("\n");
  printf("\n");
}

// test lubksb
void test_lubksb()
{
  printf("Testing lubksb...\n");
  double b[2] = {5.0, 11.0};
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;

  printf("matrix A= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }

  printf("vector b= \n");
  for (int i = 0; i < n; i++) {
    printf("%f\n", b[i]);
  }

  ludcmp(a, indx, n);
  lubksb(b, a, indx, n);
  printf("Solution vector x= \n");
  for (int i = 0; i < n; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
}

// test luminv
void test_luminv()
{
  printf("Testing luminv...\n");
  double a[4] = {1.0, 2.0, 3.0, 4.0};
  int indx[2];
  int n = 2;
  double y[4];

  printf("matrix A= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", a[i * n + j]);
    }
    printf("\n");
  }

  ludcmp(a, indx, n);
  luminv(y, a, indx, n);
  printf("Inverse matrix Y= \n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%f ", y[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

// test leastsq
void test_leastsq()
{
  printf("Testing leastsq...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;

  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  leastsq(b, a, n1, n2);
  printf("Least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("\n");
}

// test leastsq_kkt
void test_leastsq_kkt()
{
  printf("Testing leastsq_kkt...\n");
  double a[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double c[4] = {7.0, 8.0, 9.0, 10.0};
  double d[2] = {11., 15.};
  double b[3] = {7.0, 8.0, 9.0};
  int n1 = 3;
  int n2 = 2;
  int n3 = 2;
  int neq = 1;
  
  printf("Matrix A= \n");
  for (int i = 0; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  printf("Vector B= \n");
  for (int i = 0; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  printf("Matrix C= \n");
  for (int i = 0; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", c[i * n2 + j]);
    }
    printf("\n");
  }
  printf("neq = %d\n", neq);

  printf("Vector D= \n");
  for (int i = 0; i < n3; i++) {
    printf("%f\n", d[i]);
  }

  int max_iter = 20;
  int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  if (err != 0) {
    fprintf(stderr, "Error in leastsq_kkt: %d\n", err);
  }
  
  printf("Constrained least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("Number of iterations: %d\n", max_iter);
  printf("\n");
}

void test_leastsq_kkt_large()
{
  // read X matrix from file, "X.txt"
  // data size: 184x15
  printf("Testing leastsq_kkt_large...\n");

  int n1 = 184; // number of rows
  int n2 = 15; // number of columns

  double *a = (double*)malloc(n1 * n2 * sizeof(double));
  FILE *file_a = fopen("X.txt", "r");
  if (file_a == NULL) {
    fprintf(stderr, "Could not open file X.txt\n");
    return;
  }
  for (int i = 0; i < 2760; i++) {
    if (fscanf(file_a, "%lf", &a[i]) != 1) {
      fprintf(stderr, "Error reading data from file\n");
      fclose(file_a);
      return;
    }
  }
  fclose(file_a);

  double *b = (double*)malloc(n1 * sizeof(double));
  FILE *file_b = fopen("Y.txt", "r");
  if (file_b == NULL) {
    fprintf(stderr, "Could not open file b.txt\n");
    return;
  }
  for (int i = 0; i < 184; i++) {
    if (fscanf(file_b, "%lf", &b[i]) != 1) {
      fprintf(stderr, "Error reading data from file b.txt\n");
      fclose(file_b);
      return;
    }
  }
  fclose(file_b);

  // print the first 5 rows of matrix a
  printf("Matrix A :\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }
  printf("...\n");
  for (int i = n1 - 5; i < n1; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", a[i * n2 + j]);
    }
    printf("\n");
  }

  // print the first 5 rows of vector b
  printf("Vector B :\n");
  for (int i = 0; i < 5; i++) {
    printf("%f\n", b[i]);
  }
  printf("...\n");
  for (int i = n1 - 5; i < n1; i++) {
    printf("%f\n", b[i]);
  }

  int neq = 1; // number of equality constraints
  int n3 = neq + n2; // number of constraints
  double *c = (double*)malloc(n3 * n2 * sizeof(double));

  // first row: add up to 1.0
  for (int i = 0; i < n2; i++) {
    c[i] = 1.0; // equal weights
  }

  // negative identity matrix for the rest of the constraints
  for (int i = neq; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      if (i - neq == j) {
        c[i * n2 + j] = -1.0; // diagonal elements
      } else {
        c[i * n2 + j] = 0.0; // off-diagonal elements
      }
    }
  }

  double *d = (double*)malloc(n3 * sizeof(double));
  // first constraint: sum to 1.0
  d[0] = 1.0;
  // other constraints: set to 0.0
  for (int i = neq; i < n3; i++) {
    d[i] = 0.0;
  }

  // print the constraint matrix c
  printf("Constraint Matrix C (first 5 rows):\n");
  for (int i = 0; i < n3; i++) {
    for (int j = 0; j < n2; j++) {
      printf("%f ", c[i * n2 + j]);
    }
    printf("\n");
  }

  // print the constraint vector d
  printf("Constraint Vector D:\n");
  for (int i = 0; i < n3; i++) {
    printf("%f\n", d[i]);
  }

  // call leastsq_kkt
  // copy b
  double *b0 = (double*)malloc(n1 * sizeof(double));
  memcpy(b0, b, n1 * sizeof(double));

  // test solution
  double b1[15] = {0.0784, 0.1049, 0.0383, 0.1059, 0.1002, 0.0880, 0.0682, 0.0139, 0.0139, 0.0139, 0.0491, 0.0699, 0.0733, 0.0139, 0.1680};

  int max_iter = 20;
  int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
  if (err != 0) {
    fprintf(stderr, "Error in leastsq_kkt: %d\n", err);
  }

  printf("Constrained least squares solution: \n");
  for (int i = 0; i < n2; i++) {
    printf("%f\n", b[i]);
  }
  printf("Number of iterations: %d\n", max_iter);

  double cost = 0.0, cost1 = 0.0;
  for (int i = 0; i < n1; i++) {
    double diff = b0[i];
    double diff1 = b0[i];
    for (int j = 0; j < n2; j++) {
      diff -= a[i * n2 + j] * b[j];
      diff1 -= a[i * n2 + j] * b1[j];
    }
    cost += diff * diff;
    cost1 += diff1 * diff1;
  }
  printf("Cost function =  %f\n", cost);
  printf("Cost function1 =  %f\n", cost1);

  free(a);
  free(b);
  free(c);
  free(d);
  free(b0);
  printf("\n");
}

int main(int argc, char *argv[])
{
#ifdef USE_MEMORY_POOL
  poolinit();
#endif
  test_vvdot();
  test_mvdot();
  test_mmdot();
  test_ludcmp();
  test_lubksb();
  test_luminv();
  test_leastsq();
  test_leastsq_kkt();
  test_leastsq_kkt_large();
}
