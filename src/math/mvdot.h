#pragma once

#include <assert.h>

#include <configure.h>

/*!
 * \brief matrix-vector dot product: m.v
 * \param[out] r[0..n1-1] output vector
 * \param[in] m[0..n1*n2 -1] row-major sequential storage of n1 x n2 matrix
 * \param[in] v[0..n2-1] input vector
 * \param[in] n1 number of rows in matrix
 * \param[in] n2 number of columns in matrix
 */
template<typename T>
DISPATCH_MACRO void mvdot(T *r, T const *m, T const *v, int n1, int n2) {
  assert(r != v);  // r and v cannot be the same
  for (int i = 0; i < n1; ++i) {
    r[i] = 0.;
    for (int j = 0; j < n2; ++j) r[i] += m[i*n2 + j] * v[j];
  }
}
