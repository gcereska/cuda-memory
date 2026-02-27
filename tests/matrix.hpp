#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <utility>

// declaraciones so we can use the API in this functions natively
void* pmalloc(std::size_t size);
void pfree(void* p);
void* pmemcpy(void* dst, const void* src, std::size_t n);

// Matrix
class PMatrix {
 public:
  // using T = int;
  using T = double;

  PMatrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {
    const std::size_t bytes = rows_ * cols_ * sizeof(T);
    if (bytes) {
      data_ = static_cast<T*>(pmalloc(bytes));
      if (!data_) {
        std::fprintf(stderr, "[PMatrix] pmalloc(%zu) failed\n", bytes);
        std::abort();
      }
    }
  }

  // No copying i dont want to implement
  PMatrix(const PMatrix&) = delete;
  PMatrix& operator=(const PMatrix&) = delete;

  // Moves transfer ownership of the raw pointer
  PMatrix(PMatrix&& other) noexcept
      : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = other.cols_ = 0;
    other.data_ = nullptr;
  }

  PMatrix& operator=(PMatrix&& other) noexcept {
    if (this != &other) {
      release_();
      rows_ = other.rows_;
      cols_ = other.cols_;
      data_ = other.data_;
      other.rows_ = other.cols_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  ~PMatrix() { release_(); }

  std::size_t rows() const { return rows_; }          // size of rows
  std::size_t cols() const { return cols_; }          // size of cols
  std::size_t size() const { return rows_ * cols_; }  // total elements

  T* data() { return data_; }
  const T* data() const { return data_; }

  T& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
  const T& operator()(std::size_t r, std::size_t c) const {
    return data_[r * cols_ + c];
  }

  void fill_zero() {
    T* d = data_;
    for (std::size_t i = 0; i < size(); ++i) d[i] = T(0);
  }

  void fill_identity() {
    if (rows_ != cols_) {
      std::fprintf(stderr, "[PMatrix] fill_identity requires square matrix\n");
      std::abort();
    }
    fill_zero();
    for (std::size_t i = 0; i < rows_; ++i) (*this)(i, i) = T(1);
  }

  void fill_random(unsigned seed = 42, T low = -1.0, T high = 1.0) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(low, high);
    T* d = data_;
    for (std::size_t i = 0; i < size(); ++i) d[i] = dist(rng);
  }

  // C = A * B (naive i-k-j order, deterministic)
  static PMatrix multiply_naive(const PMatrix& A, const PMatrix& B) {
    if (A.cols() != B.rows()) {
      std::fprintf(
          stderr,
          "[PMatrix] multiply_naive: shape mismatch (%zu×%zu) * (%zu×%zu)\n",
          A.rows(), A.cols(), B.rows(), B.cols());
      std::abort();
    }
    PMatrix C(A.rows(), B.cols());
    C.fill_zero();
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    const T* a = A.data();
    const T* b = B.data();
    T* c = C.data();

    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t k = 0; k < K; ++k) {
        const T aik = a[i * K + k];
        const std::size_t bk = k * N;
        std::size_t ci = i * N;
        for (std::size_t j = 0; j < N; ++j) {
          c[ci + j] += aik * b[bk + j];
        }
      }
    }
    return C;
  }

  // Blocked multiply
  static PMatrix multiply_blocked(const PMatrix& A, const PMatrix& B,
                                  std::size_t BS = 128) {
    if (A.cols() != B.rows()) {
      std::fprintf(
          stderr,
          "[PMatrix] multiply_blocked: shape mismatch (%zu×%zu) * (%zu×%zu)\n",
          A.rows(), A.cols(), B.rows(), B.cols());
      std::abort();
    }
    PMatrix C(A.rows(), B.cols());
    C.fill_zero();
    const std::size_t M = A.rows(), K = A.cols(), N = B.cols();
    T* c = C.data();
    const T* a = A.data();
    const T* b = B.data();

    for (std::size_t ii = 0; ii < M; ii += BS) {
      for (std::size_t kk = 0; kk < K; kk += BS) {
        for (std::size_t jj = 0; jj < N; jj += BS) {
          const std::size_t iimax = std::min(ii + BS, M);
          const std::size_t kkmax = std::min(kk + BS, K);
          const std::size_t jjmax = std::min(jj + BS, N);
          for (std::size_t i = ii; i < iimax; ++i) {
            for (std::size_t k = kk; k < kkmax; ++k) {
              const T aik = a[i * K + k];
              T* c_row = &c[i * N + jj];
              const T* b_row = &b[k * N + jj];
              for (std::size_t j = jj; j < jjmax; ++j) {
                *c_row += aik * *b_row;
                ++c_row;
                ++b_row;
              }
            }
          }
        }
      }
    }
    return C;
  }

  // Returns max |A - B|
  static T max_abs_diff(const PMatrix& A, const PMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
      return std::numeric_limits<T>::infinity();
    T maxd = 0;
    const std::size_t n = A.size();
    const T* ad = A.data();
    const T* bd = B.data();
    for (std::size_t i = 0; i < n; ++i) {
      T d = std::fabs(ad[i] - bd[i]);
      if (d > maxd) maxd = d;
    }
    return maxd;
  }

  void release_() {
    if (data_) {
      pfree(static_cast<void*>(data_));
      data_ = nullptr;
    }
    rows_ = cols_ = 0;
  }

  std::size_t rows_ = 0, cols_ = 0;
  T* data_ = nullptr;  // raw pointer allocated via pmalloc (user ptr)
};

// find dimensions M,K,N given a target byte amount to consume
inline void choose_dims_from_budget(
    std::size_t target_bytes, std::size_t& M, std::size_t& K, std::size_t& N,
    std::size_t elem_bytes = sizeof(PMatrix::T),
    double aspect = 1.0) {  // aspect ratio -> A (M×K) × B (K×N) = C (M×N)
  // Cost will be (M*K + K*N + M*N) * elem_bytes
  double total_elems = target_bytes / double(elem_bytes);
  // Balanced case M=N=K=x -> cost = 3x^2 -> x ~ sqrt(total_elems/3)
  std::size_t x = static_cast<std::size_t>(std::sqrt(total_elems / 3.0));
  if (x < 2) x = 2;

  // Shape using aspect: M=N=x, K≈x/aspect
  M = x;
  N = x;
  K = std::max<std::size_t>(
      2, static_cast<std::size_t>(x / (aspect <= 0 ? 1.0 : aspect)));
}
