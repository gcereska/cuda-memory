#pragma once

#include <configure.h>

/*!
 * \brief vector-vector dot product: a.b
 * \param[in] a[0..n-1] first vector
 * \param[in] b[0..n-1] second vector
 */
template<typename T>
DISPATCH_MACRO T vvdot(T const *a, T const *b, int n) {
  T result = 0.;
  for (int i = 0; i < n; ++i) result += a[i] * b[i];
  return result;
}
