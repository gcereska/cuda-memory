#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <memorypool/alloc.h>
#include <memorypool/math/linalg.h>

#define NUM_TESTS 100000

void test_vol_leastsq_kkt_large() {
    int n1 = 184; // number of rows
    int n2 = 15;  // number of columns

    double *a = (double *)malloc(n1 * n2 * sizeof(double));
    FILE *file_a = fopen("X.txt", "r");
    if (file_a == NULL) {
        return;
    }
    for (int i = 0; i < 2760; i++) {
        if (fscanf(file_a, "%lf", &a[i]) != 1) {
            fclose(file_a);
            return;
        }
    }
    fclose(file_a);

    double *b = (double *)malloc(n1 * sizeof(double));
    FILE *file_b = fopen("Y.txt", "r");
    if (file_b == NULL) {
        return;
    }
    for (int i = 0; i < 184; i++) {
        if (fscanf(file_b, "%lf", &b[i]) != 1) {
            fclose(file_b);
            return;
        }
    }
    fclose(file_b);

    int neq = 1;       // number of equality constraints
    int n3 = neq + n2; // number of constraints
    double *c = (double *)malloc(n3 * n2 * sizeof(double));

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

    double *d = (double *)malloc(n3 * sizeof(double));
    // first constraint: sum to 1.0
    d[0] = 1.0;
    // other constraints: set to 0.0
    for (int i = neq; i < n3; i++) {
        d[i] = 0.0;
    }

    double *b0 = (double *)malloc(n1 * sizeof(double));
    memcpy(b0, b, n1 * sizeof(double));

    double b1[15] = {0.0784, 0.1049, 0.0383, 0.1059, 0.1002,
                     0.0880, 0.0682, 0.0139, 0.0139, 0.0139,
                     0.0491, 0.0699, 0.0733, 0.0139, 0.1680};

    int max_iter = 20;
    int err = leastsq_kkt(b, a, c, d, n1, n2, n3, neq, &max_iter);
    assert(err == 0);

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

    assert(cost > 389 && cost < 390);
    assert(cost1 > 397 && cost < 398);

    free(a);
    free(b);
    free(c);
    free(d);
    free(b0);
}

int main(int argc, char *argv[]) {
    for (int i = 0; i < NUM_TESTS; i++) {
        test_vol_leastsq_kkt_large();
    }
}
