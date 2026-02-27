// g++ -O3 allocator_bench_all.cpp pmalloc_intrusive.cpp pmalloc_binary_heap.cpp
// pmalloc_pairing_heap.cpp -o benchmark
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "matrix.hpp"
#include "pmalloc_binary_heap.hpp"
#include "pmalloc_intrusive.hpp"
#include "pmalloc_pairing_heap.hpp"

// Global function pointers
static void* (*g_pmalloc)(std::size_t) = nullptr;
static void (*g_pfree)(void*) = nullptr;
static void* (*g_pmemcpy)(void*, const void*, std::size_t) = nullptr;

void* pmalloc(std::size_t size) {
  if (g_pmalloc != nullptr) return g_pmalloc(size);
  return nullptr;
}
void pfree(void* p) {
  if (g_pfree != nullptr) g_pfree(p);
}
void* pmemcpy(void* dst, const void* src, std::size_t n) {
  if (g_pmemcpy != nullptr) return g_pmemcpy(dst, src, n);
  std::memcpy(dst, src, n);
  return dst;
}

// Test helpers
#define EXPECT_TRUE(cond, msg)         \
  do {                                 \
    if (cond)                          \
      std::printf("[PASS] %s\n", msg); \
    else {                             \
      std::printf("[FAIL] %s\n", msg); \
      std::abort();                    \
    }                                  \
  } while (0)

static double stopwatch() {
  using clock = std::chrono::high_resolution_clock;
  static auto t0 = clock::now();
  auto t = clock::now();
  return std::chrono::duration<double>(t - t0).count();
}

// using T = int;
using T = double;
static void fill_random_vals(T* a, std::size_t n, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<T> dist(-1.0, 1.0);
  std::size_t i = 0;
  while (i < n) {
    a[i] = dist(rng);
    ++i;
  }
}

static double gflops(std::size_t m, std::size_t k, std::size_t n, double s) {
  if (s <= 0.0) return 0.0;
  long double flops = 2.0L * (long double)m * (long double)k * (long double)n;
  return static_cast<double>(flops / s / 1e9L);
}

// i-k-j order (deterministic)
static void matmul_naive_raw(const T* A, const T* B, T* C, std::size_t m,
                             std::size_t k, std::size_t n) {
  std::fill(C, C + m * n, T(0));
  std::size_t i = 0;
  while (i < m) {
    const std::size_t iK = i * k;
    const std::size_t iN = i * n;
    std::size_t kk = 0;
    while (kk < k) {
      const T aik = A[iK + kk];
      const std::size_t kkN = kk * n;
      T* c = C + iN;
      const T* b = B + kkN;
      std::size_t j = 0;
      while (j < n) {
        c[j] += aik * b[j];
        ++j;
      }
      ++kk;
    }
    ++i;
  }
}

static void print_alignment(
    const char* label, void* p) {  // STILL TESTING I DO NOT FULLY UNDERSTAND
  if (p != nullptr) {
    std::uintptr_t v = reinterpret_cast<std::uintptr_t>(p);
    if ((v % 64) == 0) {
      std::printf("  %s alignment: 64B OK\n", label);
    } else {
      std::printf("  %s alignment: not 64B\n", label);
    }
  }
}

static bool equal_exact(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return false;
  std::size_t i = 0;
  while (i < a.size()) {
    if (a[i] != b[i]) return false;
    ++i;
  }
  return true;
}

static double max_abs_diff_vec(const std::vector<T>& a,
                               const std::vector<T>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  std::size_t i = 0;
  while (i < a.size()) {
    double d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
    ++i;
  }
  return m;
}

// APIs for each allocator backend
struct AllocAPI {
  const char* name;

  // optional (nullptr for libc)
  void (*init_pool_with_bytes)(std::size_t);
  void (*destroy_pool)();

  // required
  void* (*alloc)(std::size_t);
  void (*free_)(void*);
  void* (*memcpy_)(void*, const void*, std::size_t);
};

// libc adapters
static void* libc_alloc(std::size_t n) { return std::malloc(n); }
static void libc_free(void* p) { std::free(p); }
static void* libc_memcpy(void* d, const void* s, std::size_t n) {
  std::memcpy(d, s, n);
  return d;
}

// Bind and unbind API global function pointers
static void bind_matrix_to(const AllocAPI& api) {
  g_pmalloc = api.alloc;
  g_pfree = api.free_;
  g_pmemcpy = api.memcpy_;
}
static void unbind_matrix() {
  g_pmalloc = nullptr;
  g_pfree = nullptr;
  g_pmemcpy = nullptr;
}

// General Matrix Multiplication GEMM Testing
static void run_raw_compare_all(const std::vector<AllocAPI>& apis,
                                std::size_t M, std::size_t K, std::size_t N,
                                std::size_t POOL_BYTES, int REPEATS) {
  std::printf("\n===== RAW GEMM: compare all allocators =====\n");
  std::printf("Dims: M=%zu, K=%zu, N=%zu | Repeats=%d\n", M, K, N, REPEATS);

  const std::size_t bytesA = M * K * sizeof(T);
  const std::size_t bytesB = K * N * sizeof(T);
  const std::size_t bytesC = M * N * sizeof(T);

  std::vector<T> Ahost(M * K), Bhost(K * N);
  fill_random_vals(Ahost.data(), Ahost.size(), /*seed*/ 12345);
  fill_random_vals(Bhost.data(), Bhost.size(), /*seed*/ 54321);
  std::vector<std::vector<T>>
      C_all;  // each result vector C per API indexed by apii
  C_all.resize(apis.size());

  std::size_t apii = 0;
  while (apii < apis.size()) {
    const AllocAPI& api = apis[apii];
    std::printf("\n--- %s ---\n", api.name);

    if (api.init_pool_with_bytes != nullptr) {
      api.init_pool_with_bytes(POOL_BYTES);
    }

    double t0 = stopwatch();
    T* A = static_cast<T*>(api.alloc(bytesA));
    T* B = static_cast<T*>(api.alloc(bytesB));
    T* C = static_cast<T*>(api.alloc(bytesC));
    double t1 = stopwatch();

    if (A == nullptr || B == nullptr || C == nullptr) {
      std::fprintf(stderr, "[%s] allocation failed (A/B/C)\n", api.name);
      if (C != nullptr) api.free_(C);
      if (B != nullptr) api.free_(B);
      if (A != nullptr) api.free_(A);
      if (api.destroy_pool != nullptr) api.destroy_pool();
      return;
    }

    // print_alignment("A", A);
    // print_alignment("B", B);
    // print_alignment("C", C);

    double t_copy0 = stopwatch();
    std::memcpy(A, Ahost.data(), bytesA);
    std::memcpy(B, Bhost.data(), bytesB);
    double t_copy1 = stopwatch();

    int r = 1;
    double best_mm = 1e300;
    while (r <= REPEATS) {
      double t_mm0 = stopwatch();
      matmul_naive_raw(A, B, C, M, K, N);
      double t_mm1 = stopwatch();
      double tmm = t_mm1 - t_mm0;
      if (tmm < best_mm) best_mm = tmm;
      ++r;
    }

    // Host copy of C for comparison
    C_all[apii].assign(M * N, 0.0);
    double t_hcopy0 = stopwatch();
    std::memcpy(C_all[apii].data(), C, bytesC);
    double t_hcopy1 = stopwatch();

    double t_free0 = stopwatch();
    api.free_(C);
    api.free_(B);
    api.free_(A);
    double t_free1 = stopwatch();

    if (api.destroy_pool != nullptr) {  // because libc has no pool
      api.destroy_pool();
    }

    double t_alloc = t1 - t0;
    double t_copy = t_copy1 - t_copy0;
    double t_hcopy = t_hcopy1 - t_hcopy0;
    double t_free = t_free1 - t_free0;

    std::printf(
        "alloc: %.3fs | copy-in: %.3fs | gemm(best of %d): %.3fs | copy-out: "
        "%.3fs | free: %.3fs\n",
        t_alloc, t_copy, REPEATS, best_mm, t_hcopy, t_free);
    std::printf("GEMM throughput: %.2f GFLOP/s\n", gflops(M, K, N, best_mm));

    ++apii;
  }

  // Compare resulting matrixes C for each API
  if (apis.size() > 1) {
    const std::vector<T>& ref = C_all[0];
    std::size_t i = 1;
    while (i < apis.size()) {
      double maxd = max_abs_diff_vec(ref, C_all[i]);
      bool same = equal_exact(ref, C_all[i]);
      if (same == true) {
        std::printf("[OK] %s exact match with %s\n", apis[i].name,
                    apis[0].name);
      } else {
        std::printf("[MISMATCH] %s differs vs %s (max |Δ| = %.3e)\n",
                    apis[i].name, apis[0].name, maxd);
      }
      ++i;
    }
  }
}

// PMatrix: for each allocator, bind, build A,B
static void run_pmatrix_compare_all(const std::vector<AllocAPI>& apis,
                                    std::size_t M, std::size_t K, std::size_t N,
                                    std::size_t POOL_BYTES,
                                    std::size_t blockSize) {
  std::printf("\n===== PMatrix Testing =====\n");
  std::printf("Dims: M=%zu, K=%zu, N=%zu | Pool=%.1f MiB | BS=%zu\n", M, K, N,
              POOL_BYTES / (1024.0 * 1024.0), blockSize);

  // Store results for cross-allocator equality checks
  std::vector<std::vector<T>> Cnaive_all(apis.size());
  std::vector<std::vector<T>> Cblocked_all(apis.size());

  std::size_t idx = 0;
  while (idx < apis.size()) {
    const AllocAPI& api = apis[idx];
    std::printf("\n--- PMatrix: %s ---\n", api.name);

    if (api.init_pool_with_bytes != nullptr)
      api.init_pool_with_bytes(POOL_BYTES);
    bind_matrix_to(api);

    // Build A,B with fixed seeds
    PMatrix A(M, K);
    A.fill_random(111);
    PMatrix B(K, N);
    B.fill_random(222);

    // naive
    double t_n0 = stopwatch();
    PMatrix Cn = PMatrix::multiply_naive(A, B);
    double t_n1 = stopwatch();

    // Blocked
    double t_b0 = stopwatch();
    PMatrix Cb = PMatrix::multiply_blocked(A, B, blockSize);
    double t_b1 = stopwatch();

    std::printf("PMatrix naive:   %.3fs | %.2f GFLOP/s\n", t_n1 - t_n0,
                gflops(M, K, N, t_n1 - t_n0));
    std::printf("PMatrix blocked: %.3fs | %.2f GFLOP/s\n", t_b1 - t_b0,
                gflops(M, K, N, t_b1 - t_b0));

    // Copy results to host vectors
    Cnaive_all[idx].assign(Cn.data(), Cn.data() + (M * N));
    Cblocked_all[idx].assign(Cb.data(), Cb.data() + (M * N));

    unbind_matrix();
    if (api.destroy_pool != nullptr) api.destroy_pool();

    // Confirm blocked == naive per allocator (bitwise exact)
    bool same_local = equal_exact(Cnaive_all[idx], Cblocked_all[idx]);
    if (same_local == false) {
      double maxd = max_abs_diff_vec(Cnaive_all[idx], Cblocked_all[idx]);
      std::printf(
          "[MISMATCH] blocked vs naive within %s (max |Delta| = %.3e)\n",
          api.name, maxd);
    } else {
      std::printf("[OK] blocked == naive within %s (exact)\n", api.name);
    }

    ++idx;
  }

  // Cross-allocator exact equality checks (versus first)
  if (apis.size() > 1) {
    const std::vector<T>& refN = Cnaive_all[0];
    const std::vector<T>& refB = Cblocked_all[0];

    std::size_t i = 1;
    while (i < apis.size()) {
      bool sameN = equal_exact(refN, Cnaive_all[i]);
      bool sameB = equal_exact(refB, Cblocked_all[i]);
      if (!sameN) {
        double maxd = max_abs_diff_vec(refN, Cnaive_all[i]);
        std::printf("[MISMATCH] PMatrix naive %s vs %s (max |Δ| = %.3e)\n",
                    apis[i].name, apis[0].name, maxd);
      } else {
        std::printf("[OK] PMatrix naive: %s exact match with %s\n",
                    apis[i].name, apis[0].name);
      }
      if (!sameB) {
        double maxd = max_abs_diff_vec(refB, Cblocked_all[i]);
        std::printf("[MISMATCH] PMatrix blocked %s vs %s (max |Δ| = %.3e)\n",
                    apis[i].name, apis[0].name, maxd);
      } else {
        std::printf("[OK] PMatrix blocked: %s exact match with %s\n",
                    apis[i].name, apis[0].name);
      }
      ++i;
    }
  }
}

inline constexpr std::size_t KiB = 1024ULL;
inline constexpr std::size_t MiB = 1024ULL * KiB;
inline constexpr std::size_t GiB = 1024ULL * MiB;

int main() {
  // you can change these parameters
  std::size_t M = 1024, K = 1024, N = 1024;  // matrix dimensions
  std::size_t POOL_BYTES = 256 * MiB;        // pool size for pool allocators
  int REPEATS = 10;                          // repeats for raw timing
  std::size_t PMAT_BS = 128;                 // PMatrix blocked tile

  // Describe the four backends
  AllocAPI intrusive_api{
      "intrusive (pool)",           pmem_intrusive::init_pool_with_bytes,
      pmem_intrusive::destroy_pool, pmem_intrusive::pmalloc,
      pmem_intrusive::pfree,        pmem_intrusive::pmemcpy};
  AllocAPI binheap_api{
      "binary-heap (pool)",       pmem_binheap::init_pool_with_bytes,
      pmem_binheap::destroy_pool, pmem_binheap::pmalloc,
      pmem_binheap::pfree,        pmem_binheap::pmemcpy};
  AllocAPI pairing_api{
      "pairing-heap (pool)",      pmem_pairing::init_pool_with_bytes,
      pmem_pairing::destroy_pool, pmem_pairing::pmalloc,
      pmem_pairing::pfree,        pmem_pairing::pmemcpy};
  AllocAPI libc_api{"malloc/free (libc)",
                    nullptr,  // no pool init
                    nullptr,  // no pool destroy
                    libc_alloc,
                    libc_free,
                    libc_memcpy};

  std::vector<AllocAPI> apis;
  apis.push_back(intrusive_api);
  apis.push_back(binheap_api);
  apis.push_back(pairing_api);
  apis.push_back(libc_api);

  std::printf("Benchmark \n");
  std::printf(
      "Dims: M=%zu, K=%zu, N=%zu | Pool=%zu MiB | Repeats=%d | Block=%zu\n", M,
      K, N, POOL_BYTES / MiB, REPEATS, PMAT_BS);

  // time alloc/copies/GEMM per allocator
  // only does one big allocation and free per allocator
  // the bottleneck is that for some reason when allocated with our allocator
  // the actual computation is slower than with libc malloc
  run_raw_compare_all(apis, M, K, N, POOL_BYTES, REPEATS);

  // PMatrix tests with naive matrix multiplication and blocked multiplication
  // uses pfree and pmalloc internally a substantial number of times
  run_pmatrix_compare_all(apis, M, K, N, POOL_BYTES, PMAT_BS);

  std::printf("\nAll tests done.\n");
  return 0;
}
