#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/util/complex.h>
#include <ATen/native/cuda/MiscUtils.h>

#include <thrust/swap.h>

/*
  The following file contains implementation for a batched LU-factorization with partial pivoting.
  The approach is a recursive panel factorization with trailing matrix updates delegated to GEMMs/TRSMs.
  NOTE: meant as a temporary kernel before/when cuCUSOLVER/cuBLAS catches up (meant for very small matrices)
  as means to speed up the process of MAGMA deprecation while at least preserving (and even improving 1.5-2.5x)
  performance for the user on batched inputs with shapes above 256.

  Based off:

  @inproceedings{abdelfattah2019progressive,
    title={Progressive optimization of batched LU factorization on GPUs},
    author={Abdelfattah, Ahmad and Tomov, Stanimire and Dongarra, Jack},
    booktitle={2019 IEEE High Performance Extreme Computing Conference (HPEC)},
    pages={1--6},
    year={2019},
    organization={IEEE}
  }

*/


namespace at::native {

namespace {

#define LinOff(i, j, lda) i + static_cast<size_t>(j) * lda

struct LUNbConfig {
  int nb_small; // outer loop blocking factor when n < nb_crossover_n
  int nb_large; // outer loop blocking factor when n >= nb_crossover_n
};

struct LUTuning {
  int panel_threshold; // rows above this use block size (BS) 1024 tall-panel kernel
  int recnb; // recursive panel base-case width (flat column-by-column below this)
  int nb_crossover_n; // matrix size threshold: n >= this selects nb_large
  LUNbConfig nb_real; // blocking factors for float/double
  LUNbConfig nb_complex; // blocking factors for cfloat/cdouble
};

// Pre-tuned constants per compute capability
static constexpr LUTuning tuning_sm80  = {768, 10,  512, {56, 256}, {64, 256}};  // A100 (swept 2026-07-02)
static constexpr LUTuning tuning_sm89  = {768, 14,  512, {64, 384}, {96, 256}};  // L40S (swept 2026-07-05)
static constexpr LUTuning tuning_sm90  = {512, 10,  512, {40, 256}, {64, 256}};  // H100 (swept 2026-07-01)
static constexpr LUTuning tuning_sm100 = {512, 32,  512, {128, 256}, {128, 256}};  // match MAGMA nb=128 recnb=32

inline LUTuning get_tuning() {
  const auto* prop = at::cuda::getCurrentDeviceProperties();
  const auto compcap = prop->major * 10 + prop->minor;
  switch (compcap) {
    case 80: return tuning_sm80;
    case 89: return tuning_sm89;
    case 90: return tuning_sm90;
    case 100: return tuning_sm100;
    default:
      // Fallback to sm_80
      return tuning_sm80;
  };
}

// Workspace -- pointer arrays needed by cuBLAS batched TRSM + pivinfo for parallel swaps.
// pivinfo: absolute permutation vector (one per batch, size m).
template <typename scalar_t>
struct LUWorkspace {
  LUWorkspace(const Tensor& input, int nb) {
    batch_count = cuda_int_cast(batchCount(input), "batchCount");
    int m = cuda_int_cast(input.size(-2), "input.size(-2)");
    int n = cuda_int_cast(input.size(-1), "input.size(-1)");

    // Pointer arrays for cuBLAS batched TRSM (64-bit addresses)
    buffer = at::empty({2, batch_count}, input.options().dtype(at::kLong));
    dL11_array = static_cast<scalar_t**>(buffer.select(0, 0).data_ptr());
    dA12_array = static_cast<scalar_t**>(buffer.select(0, 1).data_ptr());

    // Permutation vector workspace: m ints per batch
    pivinfo_buffer = at::empty({batch_count, m}, input.options().dtype(at::kInt));
    pivinfo = static_cast<int*>(pivinfo_buffer.data_ptr());
    pivinfo_stride = m;
  }

  int batch_count;
  Tensor buffer;

  // TRSM arrays
  scalar_t** dL11_array;
  scalar_t** dA12_array;

  // Permutation workspace
  Tensor pivinfo_buffer;
  int* pivinfo;   // device pointer, batch_count * m ints
  int pivinfo_stride;  // number of rows (stride between batches)
};

// Device-side pointer array computation for TRSM.
template <typename scalar_t>
__global__ void build_trsm_ptr_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride, int lda, int batch_count,
  scalar_t** __restrict__ dL11_array,
  scalar_t** __restrict__ dA12_array,
  int diag_offset, int panel_width
) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= batch_count) return;
  auto* base = dA + b * matrix_stride;
  dL11_array[b] = base + diag_offset + static_cast<size_t>(diag_offset) * lda;
  dA12_array[b] = base + diag_offset + static_cast<size_t>(diag_offset + panel_width) * lda;
}

// TRSM + GEMM trailing-matrix update.
// Solves L11 \ A12 (TRSM), then updates A22 -= L21 @ U12 (GEMM).
// All sub-blocks are relative to (diag_offset, diag_offset) on the diagonal:
//   L11: panel_width x panel_width, unit lower triangular
//   A12: panel_width x n_right (overwritten with U12)
//   L21: m_below x panel_width
//   A22: m_below x n_right
template <typename scalar_t>
void trailing_matrix_update(
  cublasHandle_t handle,
  scalar_t* dA,
  int64_t matrix_stride,
  LUWorkspace<scalar_t>& ws,
  int lda,
  int diag_offset,
  int panel_width,
  int n_right,
  int m_below,
  int batch_count
) {
  if (n_right <= 0) return;

  // Construct TRSM scalar_t** arrays {
  int constexpr threads = 64;
  int blocks = (batch_count + threads - 1) / threads;
  build_trsm_ptr_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
    dA, matrix_stride, lda, batch_count,
    ws.dL11_array, ws.dA12_array,
    diag_offset, panel_width
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // }

  auto constexpr one = static_cast<scalar_t>(1);
  auto constexpr neg_one = static_cast<scalar_t>(-1);
  at::cuda::blas::trsmBatched<scalar_t>(
    handle,
    CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
    CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
    panel_width, n_right, &one,
    ws.dL11_array, lda,
    ws.dA12_array, lda,
    batch_count
  );

  if (m_below > 0) {
    size_t off_L21 = (diag_offset + panel_width) + static_cast<size_t>(diag_offset) * lda;
    size_t off_U12 = diag_offset + static_cast<size_t>(diag_offset + panel_width) * lda;
    size_t off_A22 = (diag_offset + panel_width) + static_cast<size_t>(diag_offset + panel_width) * lda;

    at::cuda::blas::bgemm(
      'n', 'n',
      m_below, n_right, panel_width,
      neg_one,
      dA + off_L21, lda, matrix_stride,
      dA + off_U12, lda, matrix_stride,
      one,
      dA + off_A22, lda, matrix_stride,
      batch_count
    );
  }
}

// Argmax Abs helpers {
template <typename real_t>
__device__ __forceinline__ void warp_argmax(real_t& val, int& idx) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    real_t other_val = __shfl_down_sync(0xffffffff, val, offset);
    int    other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename real_t, int BS>
__device__ __forceinline__ int block_argmax(
  real_t my_max, int my_idx,
  real_t* sdata, int* sidx, int tid
) {
  warp_argmax(my_max, my_idx);
  int warp_id = tid / 32;
  int lane = tid % 32;

  if (lane == 0) {
    sdata[warp_id] = my_max;
    sidx[warp_id] = my_idx;
  }
  __syncthreads();

  constexpr auto NWARPS = BS / 32;
  if (tid < 32) {
    auto v = (tid < NWARPS) ? sdata[tid] : static_cast<real_t>(-1);
    auto i = (tid < NWARPS) ? sidx[tid] : -1;
    warp_argmax(v, i);
    if (tid == 0) {
      sidx[0] = i;
    }
  }
  __syncthreads();

  return sidx[0];
}
// }

// Convert LAPACK-style sequential swap ipiv into an absolute permutation vector.
// After this kernel, pivinfo[i] (0-based) gives the source row for destination
// row (row_offset + i). Only rows [row_offset, row_offset + nrows) participate.
//
// Algorithm (same as MAGMA's setup_pivinfo_devfunc):
//   1. All threads initialize pivinfo as identity: pivinfo[i] = row_offset + i
//   2. Thread 0 replays the nb swaps sequentially on the identity.
//
// Launch: one block per batch, blockDim.x >= nrows (or loop if nrows > BS).
template <int BS>
__global__ void __launch_bounds__(BS)
setup_pivinfo_kernel(
  int* __restrict__ pivinfo,    // output: [batch_count, pivinfo_stride]
  int pivinfo_stride,           // stride between batches in pivinfo
  const int* __restrict__ ipiv, // input: LAPACK pivot indices (1-based)
  int ipiv_stride,              // stride between batches in ipiv
  int row_offset,               // first row index (= col_start)
  int nrows,                    // number of rows in submatrix (= m - col_start)
  int nb                        // number of pivots to replay
) {
  int batch = blockIdx.x;
  int tid = threadIdx.x;

  int* piv = pivinfo + batch * pivinfo_stride;
  const int* ip = ipiv + batch * ipiv_stride;

  // Initialize identity (1-based absolute row indices, like MAGMA)
  for (int dst = tid + row_offset; dst < row_offset + nrows; dst += BS) {
    piv[dst] = dst + 1;
  }
  __syncthreads();

  // Thread 0 replays the sequential swaps
  if (tid == 0) {
    for (int src = row_offset; src < row_offset + nb; ++src) {
      auto dst = ip[src] - 1;
      if (src != dst) {
        thrust::swap(piv[src], piv[dst]);
      }
    }
  }
}

void setup_pivinfo(
  int m,
  int col_start,
  int nb,
  const int* dipiv,
  int ipiv_stride,
  int* dpivinfo,
  int pivinfo_stride,
  int batch_count
) {
  int nrows = m - col_start;
  int constexpr BS = 256;
  setup_pivinfo_kernel<BS><<<batch_count, BS, 0, at::cuda::getCurrentCUDAStream()>>>(
    dpivinfo, pivinfo_stride,
    dipiv, ipiv_stride,
    col_start, nrows, nb
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Row-parallel swap: similar to MAGMA's dlaswp_rowparallel_devfunc.
// nb threads, each handles one row. Gathers source row into shared memory (strided),
// patches dA, then copies from shared memory into dA (coalesced).
// Direct swaps inflict strided reads/writes.
// pivinfo is 1-based. blockDim.x = nb (= height). Tiles across columns via grid.x.
template <typename scalar_t>
__global__ void
laswp_rowparallel_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda,
  const int* __restrict__ pivinfo, // [batch_count, pivinfo_stride], 1-based
  int pivinfo_stride,
  int row_offset,   // = col_start
  int nb,           // number of rows = height = blockDim.x
  int ncols,        // total columns
  int col_offset,   // first column (absolute)
  int swp_width     // columns per tile
) {
  extern __shared__ char smem_raw[];
  scalar_t* sdata = reinterpret_cast<scalar_t*>(smem_raw);

  int batch = blockIdx.z;
  int tid = threadIdx.x;

  auto* A = dA + batch * matrix_stride;
  const int* piv = pivinfo + batch * pivinfo_stride + row_offset;

  // This tile's column range
  int tile_col_start = blockIdx.x * swp_width;
  int tile_width = ::min(swp_width, ncols - tile_col_start);

  if (tid < nb) {
    // src/dst rows
    int src = piv[tid] - 1;
    int dst = piv[src - row_offset] - 1;

    // Pass 1: gather source into shared memory, patch dA.
    // Strided read/write.
    for (int i = 0; i < tile_width; ++i) {
      int col = col_offset + tile_col_start + i;
      sdata[tid + i * nb] = A[LinOff(src, col, lda)];
      A[LinOff(src, col, lda)] = A[LinOff(dst, col, lda)];
    }
  }
  __syncthreads();

  if (tid < nb) {
    // Pass 2: write shared memory back -- coalesced write
    auto row = row_offset + tid;
    for (int i = 0; i < tile_width; ++i) {
      auto col = col_offset + tile_col_start + i;
      A[LinOff(row, col, lda)] = sdata[tid + i * nb];
    }
  }
}

// Parallel pivot application using permutation vector.
// Gathers permuted rows into shared memory (strided access),
// then copies them back (coalesced write).
// Direct swaps inflict strided reads and writes.
template <typename scalar_t>
void batched_apply_pivots_parallel(
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int col_start,
  int nb,
  const int* dipiv,
  int ipiv_stride,
  LUWorkspace<scalar_t>& ws,
  int col_lo,
  int col_hi,
  int batch_count
) {
  auto ncols = col_hi - col_lo;
  if (ncols <= 0 || nb <= 0) return;

  // Small tile width for high occupancy (matches MAGMA's SWP_WIDTH=4)
  int swp_width = std::min(4, ncols);
  int col_tiles = (ncols + swp_width - 1) / swp_width;
  size_t shmem = nb * swp_width * sizeof(scalar_t);
  auto grid = dim3(col_tiles, 1, batch_count);

  laswp_rowparallel_kernel<scalar_t><<<grid, nb, shmem, at::cuda::getCurrentCUDAStream()>>>(
    dA, matrix_stride, lda,
    ws.pivinfo, ws.pivinfo_stride,
    col_start, nb,
    ncols, col_lo, swp_width
  );
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Register-resident fused panel factorization (similar to MAGMA's sgetf2_fused_device).
// Each thread owns one row of the panel in registers (rA[WIDTH]).
// Pivot search via shared-memory parallel reduction, virtual row swap via rowid tracking,
// in-register scale and rank-1 update. One global read at start, one write at end.
// blockDim.x = nrows (number of rows in the submatrix), one block per batch.
// Constraint: nrows <= 1024 (max threads per block).
template <typename scalar_t, int WIDTH>
__global__ void
batched_panel_fused_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda, int m,
  int col_start,
  int ipiv_stride,
  int* __restrict__ dipiv,
  int* __restrict__ dinfo
) {
  using real_t = c10::scalar_value_type<scalar_t>::type;

  const int tx = threadIdx.x;
  const int batch = blockIdx.x;
  const int nrows = m - col_start;

  auto* A = dA + batch * matrix_stride;

  // Shared memory layout:
  //   sx[WIDTH]         - pivot row values (for broadcast)
  //   dsx[nrows]        - absolute values for reduction
  //   isx[nrows]        - indices for reduction
  //   sipiv[WIDTH]      - pivot indices
  extern __shared__ char smem_raw[];
  scalar_t* sx = reinterpret_cast<scalar_t*>(smem_raw);
  real_t* dsx = reinterpret_cast<real_t*>(sx + WIDTH);
  int* isx = reinterpret_cast<int*>(dsx + nrows);
  int* sipiv = reinterpret_cast<int*>(isx + nrows);

  // Each thread loads its full row into registers
  scalar_t rA[WIDTH];
  #pragma unroll
  for (int i = 0; i < WIDTH; i++) {
    rA[i] = (tx < nrows) ? A[LinOff(col_start + tx, col_start + i, lda)] : static_cast<scalar_t>(0);
  }

  int rowid = tx;
  int linfo = (col_start == 0) ? 0 : dinfo[batch];

  if (tx < WIDTH) {
    sipiv[tx] = 0;
  }

  for (int i = 0; i < WIDTH; i++) {
    // 1. Write abs value to shared memory using current logical row position
    dsx[rowid] = std::abs(rA[i]);
    isx[tx] = tx;
    __syncthreads();

    // 2. Parallel reduction for argmax over rows [i, nrows)
    // Cross-warp steps via shared memory
    if (nrows - i > 512) { if (tx < 512 && tx + 512 < nrows - i) { if (dsx[i+tx] < dsx[i+tx+512] || (dsx[i+tx] == dsx[i+tx+512] && isx[i+tx+512] < isx[i+tx])) { dsx[i+tx] = dsx[i+tx+512]; isx[i+tx] = isx[i+tx+512]; } } __syncthreads(); }
    if (nrows - i > 256) { if (tx < 256 && tx + 256 < nrows - i) { if (dsx[i+tx] < dsx[i+tx+256] || (dsx[i+tx] == dsx[i+tx+256] && isx[i+tx+256] < isx[i+tx])) { dsx[i+tx] = dsx[i+tx+256]; isx[i+tx] = isx[i+tx+256]; } } __syncthreads(); }
    if (nrows - i > 128) { if (tx < 128 && tx + 128 < nrows - i) { if (dsx[i+tx] < dsx[i+tx+128] || (dsx[i+tx] == dsx[i+tx+128] && isx[i+tx+128] < isx[i+tx])) { dsx[i+tx] = dsx[i+tx+128]; isx[i+tx] = isx[i+tx+128]; } } __syncthreads(); }
    if (nrows - i >  64) { if (tx <  64 && tx +  64 < nrows - i) { if (dsx[i+tx] < dsx[i+tx+ 64] || (dsx[i+tx] == dsx[i+tx+ 64] && isx[i+tx+ 64] < isx[i+tx])) { dsx[i+tx] = dsx[i+tx+ 64]; isx[i+tx] = isx[i+tx+ 64]; } } __syncthreads(); }
    if (nrows - i >  32) { if (tx <  32 && tx +  32 < nrows - i) { if (dsx[i+tx] < dsx[i+tx+ 32] || (dsx[i+tx] == dsx[i+tx+ 32] && isx[i+tx+ 32] < isx[i+tx])) { dsx[i+tx] = dsx[i+tx+ 32]; isx[i+tx] = isx[i+tx+ 32]; } } __syncthreads(); }
    // Warp-level reduction for final 32 elements
    real_t rx_abs_max;
    int max_id;
    if (tx < 32) {
      real_t val = (tx < nrows - i) ? dsx[i + tx] : static_cast<real_t>(-1);
      int idx = (tx < nrows - i) ? isx[i + tx] : tx;
      unsigned mask = 0xffffffff;
      #pragma unroll
      for (int s = 16; s >= 1; s >>= 1) {
        real_t other_val = __shfl_down_sync(mask, val, s);
        int other_idx = __shfl_down_sync(mask, idx, s);
        if (other_val > val || (other_val == val && other_idx < idx)) { val = other_val; idx = other_idx; }
      }
      if (tx == 0) { dsx[i] = val; isx[i] = idx; }
    }
    __syncthreads();
    rx_abs_max = dsx[i];
    max_id = isx[i];

    linfo = (rx_abs_max == static_cast<real_t>(0) && linfo == 0) ? (col_start + i + 1) : linfo;

    if (tx == 0) {
      sipiv[i] = max_id;
    }
    __syncthreads();

    // 3. Pivot row broadcasts its values to shared memory
    if (rowid == max_id) {
      #pragma unroll
      for (int j = 0; j < WIDTH; j++) {
        sx[j] = rA[j];
      }
    }
    __syncthreads();

    // 4. Virtual row swap
    if (rx_abs_max != static_cast<real_t>(0)) {
      if (rowid == max_id) {
        rowid = i;
      } else if (rowid == i) {
        rowid = max_id;
      }
    }
    __syncthreads();

    // 5. Scale and rank-1 update (in registers)
    scalar_t reg = (rx_abs_max == static_cast<real_t>(0))
        ? static_cast<scalar_t>(1)
        : static_cast<scalar_t>(1) / sx[i];

    if (rowid > i) {
      rA[i] *= reg;
      #pragma unroll
      for (int j = i + 1; j < WIDTH; j++) {
        rA[j] -= rA[i] * sx[j];
      }
    }
  }

  // Write info
  if (tx == 0) {
    dinfo[batch] = linfo;
  }

  // Write pivots (1-based, absolute)
  if (tx < WIDTH) {
    dipiv[batch * ipiv_stride + col_start + tx] = sipiv[tx] + col_start + 1;
  }

  // Write back results using remapped rowid
  if (tx < nrows) {
    #pragma unroll
    for (int i = 0; i < WIDTH; i++) {
      A[LinOff(col_start + rowid, col_start + i, lda)] = rA[i];
    }
  }
}

// Dispatch helper for register-resident fused panel kernel (WIDTH 1-32)
template <typename scalar_t>
bool try_launch_fused_panel(
  scalar_t* dA, int64_t matrix_stride, int lda, int m,
  int col_start, int nb,
  int* dipiv, int ipiv_stride,
  int* dinfo, int batch_count
) {
  int nrows = m - col_start;
  // Fused kernel needs one thread per row, max 1024
  if (nrows > 1024 || nb > 32) return false;

  // Shared memory: WIDTH * sizeof(scalar_t) + nrows * sizeof(real_t) + nrows * sizeof(int) + WIDTH * sizeof(int)
  using real_t = c10::scalar_value_type<scalar_t>::type;
  size_t shmem = nb * sizeof(scalar_t) + nrows * sizeof(real_t) + nrows * sizeof(int) + nb * sizeof(int);

  dim3 grid(batch_count);
  dim3 threads(nrows);

  auto stream = at::cuda::getCurrentCUDAStream();

  #define LAUNCH_FUSED(W) \
    batched_panel_fused_kernel<scalar_t, W><<<grid, threads, shmem, stream>>>( \
      dA, matrix_stride, lda, m, col_start, ipiv_stride, dipiv, dinfo)

  switch (nb) {
    case  1: LAUNCH_FUSED( 1); break;
    case  2: LAUNCH_FUSED( 2); break;
    case  3: LAUNCH_FUSED( 3); break;
    case  4: LAUNCH_FUSED( 4); break;
    case  5: LAUNCH_FUSED( 5); break;
    case  6: LAUNCH_FUSED( 6); break;
    case  7: LAUNCH_FUSED( 7); break;
    case  8: LAUNCH_FUSED( 8); break;
    case  9: LAUNCH_FUSED( 9); break;
    case 10: LAUNCH_FUSED(10); break;
    case 11: LAUNCH_FUSED(11); break;
    case 12: LAUNCH_FUSED(12); break;
    case 13: LAUNCH_FUSED(13); break;
    case 14: LAUNCH_FUSED(14); break;
    case 15: LAUNCH_FUSED(15); break;
    case 16: LAUNCH_FUSED(16); break;
    case 17: LAUNCH_FUSED(17); break;
    case 18: LAUNCH_FUSED(18); break;
    case 19: LAUNCH_FUSED(19); break;
    case 20: LAUNCH_FUSED(20); break;
    case 21: LAUNCH_FUSED(21); break;
    case 22: LAUNCH_FUSED(22); break;
    case 23: LAUNCH_FUSED(23); break;
    case 24: LAUNCH_FUSED(24); break;
    case 25: LAUNCH_FUSED(25); break;
    case 26: LAUNCH_FUSED(26); break;
    case 27: LAUNCH_FUSED(27); break;
    case 28: LAUNCH_FUSED(28); break;
    case 29: LAUNCH_FUSED(29); break;
    case 30: LAUNCH_FUSED(30); break;
    case 31: LAUNCH_FUSED(31); break;
    case 32: LAUNCH_FUSED(32); break;
    default: return false;
  }
  #undef LAUNCH_FUSED

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

// cuBLAS Batched TRSM only works with square inputs,
// hence the need for this kernel for rectangular inputs
template <typename scalar_t, int BS>
__global__ void __launch_bounds__(BS)
batched_panel_colserial_fused_kernel(
  scalar_t* __restrict__ dA, int64_t matrix_stride,
  int lda, int m,
  int col_start, int nb,
  int ipiv_stride,
  int* __restrict__ dipiv,
  int* __restrict__ dinfo
) {
  using real_t = c10::scalar_value_type<scalar_t>::type;

  int constexpr NWARPS = BS / 32;
  __shared__ real_t sdata[NWARPS];
  __shared__ int sidx[NWARPS];
  __shared__ scalar_t sdiag;

  int batch = blockIdx.z;
  auto* A = dA + batch * matrix_stride;
  int tid = threadIdx.x;
  int panel_end = col_start + nb;

  for (int k = col_start; k < panel_end; ++k) {
    int rows_below = m - k - 1;
    int update_cols = panel_end - k - 1;

    // 1. Pivot find (warp-shuffle reduction)
    auto my_max = static_cast<real_t>(-1);
    auto my_idx = -1;
    for (int i = k + tid; i < m; i += BS) {
      auto v = std::abs(A[LinOff(i, k, lda)]);
      if (v > my_max) {
        my_max = v;
        my_idx = i;
      }
    }
    int pivot_row = block_argmax<real_t, BS>(my_max, my_idx, sdata, sidx, tid);
    if (tid == 0) {
      dipiv[batch * ipiv_stride + k] = pivot_row + 1; // 1-based!
    }

    // 2. Row swaps
    if (pivot_row != k) {
      for (int j = tid + col_start; j < nb + col_start; j += BS) {
        auto src = LinOff(k, j, lda);
        auto dst = LinOff(pivot_row, j, lda);
        thrust::swap(A[src], A[dst]);
      }
    }
    __syncthreads();

    // 3. Scale (divide by diagonal - skip if zero for singular matrices)
    if (tid == 0) {
      sdiag = A[LinOff(k, k, lda)];
      if (std::abs(sdiag) == 0 && dinfo[batch] == 0) {
        dinfo[batch] = k + 1; // 1-based!
      }
    }
    __syncthreads();

    if (std::abs(sdiag) != 0) {
      for (int i = k + 1 + tid; i < m; i += BS) {
        A[LinOff(i, k, lda)] /= sdiag;
      }
    }
    __syncthreads();

    // 4. Rank-1 update (linearized)
    if (rows_below > 0 && update_cols > 0) {
      auto numel = rows_below * update_cols;
      for (int idx = tid; idx < numel; idx += BS) {
        auto local_row = idx % rows_below;
        auto local_col = idx / rows_below;
        auto i = k + 1 + local_row;
        auto j = k + 1 + local_col;
        A[LinOff(i, j, lda)] -= A[LinOff(i, k, lda)] * A[LinOff(k, j, lda)];
      }
    }
  } // for cols in the panel
}

template <typename scalar_t>
void lu_batched_panel_recursive(
  cublasHandle_t handle,
  scalar_t* dA,
  int64_t matrix_stride,
  int lda,
  int m,
  int col_start,
  int nb,
  int* dipiv,
  int ipiv_stride,
  int* dinfo,
  int batch_count,
  LUWorkspace<scalar_t>& ws,
  const LUTuning& tuning
) {
  // Base case: use fused register-resident panel if possible, else fall back
  if (nb <= tuning.recnb) {
    if (try_launch_fused_panel<scalar_t>(
          dA, matrix_stride, lda, m,
          col_start, nb, dipiv, ipiv_stride, dinfo, batch_count)) {
      return;
    }
    // Fallback: nrows > 1024 or nb > 32
    auto grid = dim3(1, 1, batch_count);
    if ((m - col_start) > tuning.panel_threshold) {
      batched_panel_colserial_fused_kernel<scalar_t, 1024><<<grid, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        dA, matrix_stride, lda, m,
        col_start, nb,
        ipiv_stride, dipiv, dinfo
      );
    } else {
      batched_panel_colserial_fused_kernel<scalar_t, 256><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        dA, matrix_stride, lda, m,
        col_start, nb,
        ipiv_stride, dipiv, dinfo
      );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto n1 = nb / 2;
  auto n2 = nb - n1;

  // 1. Factor left half: columns [col_start, col_start + n1)
  lu_batched_panel_recursive<scalar_t>(
    handle,
    dA, matrix_stride, lda, m,
    col_start, n1,
    dipiv, ipiv_stride, dinfo,
    batch_count, ws, tuning
  );

  // 2. Apply left-half pivots to right half columns [col_start + n1, col_start + nb)
  setup_pivinfo(m, col_start, n1, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
  batched_apply_pivots_parallel<scalar_t>(
    dA, matrix_stride, lda, m,
    col_start, n1,
    dipiv, ipiv_stride,
    ws,
    col_start + n1, col_start + nb, batch_count
  );

  // 3. TRSM + GEMM: trailing update
  trailing_matrix_update<scalar_t>(
    handle, dA, matrix_stride, ws, lda,
    col_start, n1, n2, m - col_start - n1, batch_count
  );

  // 4. Factor right half: columns [col_start + n1, col_start + nb)
  lu_batched_panel_recursive<scalar_t>(
    handle,
    dA, matrix_stride, lda, m,
    col_start + n1, n2,
    dipiv, ipiv_stride, dinfo,
    batch_count, ws, tuning
  );

  // 5. Apply right-half pivots back to left half columns [col_start, col_start + n1)
  setup_pivinfo(m, col_start + n1, n2, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
  batched_apply_pivots_parallel<scalar_t>(
    dA, matrix_stride, lda, m,
    col_start + n1, n2,
    dipiv, ipiv_stride,
    ws,
    col_start, col_start + n1, batch_count
  );
}

} // anonymous namespace

void lu_batched_blas3_kernel(const Tensor& input, const Tensor& pivots, const Tensor& infos) {
  const auto tuning = get_tuning();
  int batch_count = cuda_int_cast(batchCount(input), "batchCount");
  int m = cuda_int_cast(input.size(-2), "input.size(-2)");
  int n = cuda_int_cast(input.size(-1), "input.size(-1)");
  int64_t matrix_stride = matrixStride(input);
  int lda = std::max(cuda_int_cast(input.stride(-1), "input.stride(-1)"), std::max(1, m));

  NoTF32Guard disable_tf32;
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  infos.zero_();

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(input.scalar_type(), "linalg_lu_batched_blas3_kernel", [&] {
    auto* dA = static_cast<scalar_t*>(input.data_ptr());
    auto* dipiv = static_cast<int*>(pivots.data_ptr());
    auto* dinfo = static_cast<int*>(infos.data_ptr());

    LUNbConfig nbc;
    if constexpr (c10::is_complex<scalar_t>::value) {
      nbc = tuning.nb_complex;
    } else {
      nbc = tuning.nb_real;
    }

    int nb = (n >= tuning.nb_crossover_n) ? nbc.nb_large : nbc.nb_small;
    auto ws = LUWorkspace<scalar_t>(input, nb);
    auto min_mn = std::min(m, n);
    auto ipiv_stride = min_mn;

    // Right-looking blocked LU: step through columns in blocks of nb.
    // Each iteration factors one panel of width actual_nb, then updates the
    // trailing matrix to the right.
    // The panel itself is factored recursively (splitting its width in half
    // down to recnb, same algorithm as MAGMA's dgetrf_recpanel_batched).
    for (int j = 0; j < min_mn; j += nb) {
      auto actual_nb = std::min(nb, min_mn - j);

      // 1. Panel factorization
      lu_batched_panel_recursive<scalar_t>(
        handle,
        dA, matrix_stride, lda, m,
        j, actual_nb,
        dipiv, ipiv_stride, dinfo,
        batch_count, ws, tuning
      );

      // 2. Propagate pivots to columns outside the panel (row-parallel)
      //    Left side: cols [0, j)
      setup_pivinfo(m, j, actual_nb, dipiv, ipiv_stride, ws.pivinfo, ws.pivinfo_stride, batch_count);
      batched_apply_pivots_parallel<scalar_t>(
        dA, matrix_stride, lda, m,
        j, actual_nb,
        dipiv, ipiv_stride,
        ws,
        0, j, batch_count
      );
      //    Right side: cols [j + actual_nb, n)
      batched_apply_pivots_parallel<scalar_t>(
        dA, matrix_stride, lda, m,
        j, actual_nb,
        dipiv, ipiv_stride,
        ws,
        j + actual_nb, n, batch_count
      );

      // 3. Trailing matrix update
      trailing_matrix_update<scalar_t>(
        handle, dA, matrix_stride, ws, lda,
        j, actual_nb, n - j - actual_nb, m - j - actual_nb, batch_count
      );
    }
  });
}

} // at::native
