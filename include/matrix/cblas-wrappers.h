// matrix/cblas-wrappers.h

// Copyright 2021 NVIDIA

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_MATRIX_CBLAS_WRAPPERS_H_
#define KALDI_MATRIX_CBLAS_WRAPPERS_H_ 1


#include <limits>
#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"
// #include "matrix/kaldi-blas.h"
// #include "matrix/matrix-common.h"

// Do not include this file directly.  It is to be included
// by .cc files in this directory.

typedef int32_t KaldiBlasInt;

enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

namespace kaldi {

typedef int32_t MatrixIndexT;

// typedef enum {
//   kTrans    = 112, // = CblasTrans
//   kNoTrans  = 111  // = CblasNoTrans
// } MatrixTransposeType;

inline void cblas_Xcopy(const int N, const float *X, const int incX, float *Y,
                        const int incY) {
    assert(0 && "dummy cblas library has no implementations");
}

inline void cblas_Xcopy(const int N, const double *X, const int incX, double *Y,
                        const int incY) {
    assert(0 && "dummy cblas library has no implementations");
}


inline float cblas_Xasum(const int N, const float *X, const int incX) {
    assert(0 && "dummy cblas library has no implementations");
    return 0;
}

inline double cblas_Xasum(const int N, const double *X, const int incX) {
    assert(0 && "dummy cblas library has no implementations");
    return 0;
}

inline void cblas_Xrot(const int N, float *X, const int incX, float *Y,
                       const int incY, const float c, const float s) {
    assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xrot(const int N, double *X, const int incX, double *Y,
                       const int incY, const double c, const double s) {
    assert(0 && "dummy cblas library has no implementations");
}
inline float cblas_Xdot(const int N, const float *const X,
                        const int incX, const float *const Y,
                        const int incY) {
    assert(0 && "dummy cblas library has no implementations");
    return 0;
}
inline double cblas_Xdot(const int N, const double *const X,
                        const int incX, const double *const Y,
                        const int incY) {
    assert(0 && "dummy cblas library has no implementations");
    return 0;
}
inline void cblas_Xaxpy(const int N, const float alpha, const float *X,
                        const int incX, float *Y, const int incY) {
    assert(0 && "dummy cblas library has no implementations");  
}
inline void cblas_Xaxpy(const int N, const double alpha, const double *X,
                        const int incX, double *Y, const int incY) {
    assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xscal(const int N, const float alpha, float *data,
                        const int inc) {
    assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xscal(const int N, const double alpha, double *data, 
                        const int inc) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xspmv(const float alpha, const int num_rows, const float *Mdata,
                        const float *v, const int v_inc,
                        const float beta, float *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xspmv(const double alpha, const int num_rows, const double *Mdata,
                        const double *v, const int v_inc,
                        const double beta, double *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xtpmv(MatrixTransposeType trans, const float *Mdata,
                        const int num_rows, float *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xtpmv(MatrixTransposeType trans, const double *Mdata,
                        const int num_rows, double *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}


inline void cblas_Xtpsv(MatrixTransposeType trans, const float *Mdata,
                        const int num_rows, float *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xtpsv(MatrixTransposeType trans, const double *Mdata,
                        const int num_rows, double *y, const int y_inc) {
  assert(0 && "dummy cblas library has no implementations");
}

// x = alpha * M * y + beta * x
inline void cblas_Xspmv(MatrixIndexT dim, float alpha, const float *Mdata,
                        const float *ydata, MatrixIndexT ystride,
                        float beta, float *xdata, MatrixIndexT xstride) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xspmv(MatrixIndexT dim, double alpha, const double *Mdata,
                        const double *ydata, MatrixIndexT ystride,
                        double beta, double *xdata, MatrixIndexT xstride) {
  assert(0 && "dummy cblas library has no implementations");    
}

// Implements  A += alpha * (x y'  + y x'); A is symmetric matrix.
inline void cblas_Xspr2(MatrixIndexT dim, float alpha, const float *Xdata,
                        MatrixIndexT incX, const float *Ydata, MatrixIndexT incY,
                          float *Adata) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xspr2(MatrixIndexT dim, double alpha, const double *Xdata,
                        MatrixIndexT incX, const double *Ydata, MatrixIndexT incY,
                        double *Adata) {
  assert(0 && "dummy cblas library has no implementations");
}

// Implements  A += alpha * (x x'); A is symmetric matrix.
inline void cblas_Xspr(MatrixIndexT dim, float alpha, const float *Xdata,
                       MatrixIndexT incX, float *Adata) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xspr(MatrixIndexT dim, double alpha, const double *Xdata,
                       MatrixIndexT incX, double *Adata) {
  assert(0 && "dummy cblas library has no implementations");
}

// sgemv,dgemv: y = alpha M x + beta y.
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, float alpha, const float *Mdata,
                        MatrixIndexT stride, const float *xdata,
                        MatrixIndexT incX, float beta, float *ydata, MatrixIndexT incY) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xgemv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, double alpha, const double *Mdata,
                        MatrixIndexT stride, const double *xdata,
                        MatrixIndexT incX, double beta, double *ydata, MatrixIndexT incY) {
  assert(0 && "dummy cblas library has no implementations");
}

// sgbmv, dgmmv: y = alpha M x +  + beta * y.
inline void cblas_Xgbmv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, MatrixIndexT num_below,
                        MatrixIndexT num_above, float alpha, const float *Mdata,
                        MatrixIndexT stride, const float *xdata,
                        MatrixIndexT incX, float beta, float *ydata, MatrixIndexT incY) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xgbmv(MatrixTransposeType trans, MatrixIndexT num_rows,
                        MatrixIndexT num_cols, MatrixIndexT num_below,
                        MatrixIndexT num_above, double alpha, const double *Mdata,
                        MatrixIndexT stride, const double *xdata,
                        MatrixIndexT incX, double beta, double *ydata, MatrixIndexT incY) {
  assert(0 && "dummy cblas library has no implementations");
}


template<typename Real>
inline void Xgemv_sparsevec(MatrixTransposeType trans, MatrixIndexT num_rows,
                            MatrixIndexT num_cols, Real alpha, const Real *Mdata,
                            MatrixIndexT stride, const Real *xdata,
                            MatrixIndexT incX, Real beta, Real *ydata,
                            MatrixIndexT incY) {
  assert(0 && "dummy cblas library has no implementations");
}

inline void cblas_Xgemm(const float alpha,
                        MatrixTransposeType transA,
                        const float *Adata,
                        MatrixIndexT a_num_rows, MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                        MatrixTransposeType transB, 
                        const float *Bdata, MatrixIndexT b_stride,
                        const float beta,
                        float *Mdata, 
                        MatrixIndexT num_rows, MatrixIndexT num_cols,MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xgemm(const double alpha,
                        MatrixTransposeType transA,
                        const double *Adata,
                        MatrixIndexT a_num_rows, MatrixIndexT a_num_cols, MatrixIndexT a_stride,
                        MatrixTransposeType transB, 
                        const double *Bdata, MatrixIndexT b_stride,
                        const double beta,
                        double *Mdata, 
                        MatrixIndexT num_rows, MatrixIndexT num_cols,MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}


inline void cblas_Xsymm(const float alpha,
                        MatrixIndexT sz,
                        const float *Adata,MatrixIndexT a_stride,
                        const float *Bdata,MatrixIndexT b_stride,
                        const float beta,
                        float *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xsymm(const double alpha,
                        MatrixIndexT sz,
                        const double *Adata,MatrixIndexT a_stride,
                        const double *Bdata,MatrixIndexT b_stride,
                        const double beta,
                        double *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}
// ger: M += alpha x y^T.
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, float alpha,
                       const float *xdata, MatrixIndexT incX, const float *ydata,
                       MatrixIndexT incY, float *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void cblas_Xger(MatrixIndexT num_rows, MatrixIndexT num_cols, double alpha,
                       const double *xdata, MatrixIndexT incX, const double *ydata,
                       MatrixIndexT incY, double *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}

// syrk: symmetric rank-k update.
// if trans==kNoTrans, then C = alpha A A^T + beta C
// else C = alpha A^T A + beta C.
// note: dim_c is dim(C), other_dim_a is the "other" dimension of A, i.e.
// num-cols(A) if kNoTrans, or num-rows(A) if kTrans.
// We only need the row-major and lower-triangular option of this, and this
// is hard-coded.
inline void cblas_Xsyrk (
    const MatrixTransposeType trans, const MatrixIndexT dim_c,
    const MatrixIndexT other_dim_a, const float alpha, const float *A,
    const MatrixIndexT a_stride, const float beta, float *C,
    const MatrixIndexT c_stride) {
  assert(0 && "dummy cblas library has no implementations");
}

inline void cblas_Xsyrk(
    const MatrixTransposeType trans, const MatrixIndexT dim_c,
    const MatrixIndexT other_dim_a, const double alpha, const double *A,
    const MatrixIndexT a_stride, const double beta, double *C,
    const MatrixIndexT c_stride) {
  assert(0 && "dummy cblas library has no implementations");
}

/// matrix-vector multiply using a banded matrix; we always call this
/// with b = 1 meaning we're multiplying by a diagonal matrix.  This is used for
/// elementwise multiplication.  We miss some of the arguments out of this
/// wrapper.
inline void cblas_Xsbmv1(
    const MatrixIndexT dim,
    const double *A,
    const double alpha,
    const double *x,
    const double beta,
    double *y) {
  assert(0 && "dummy cblas library has no implementations");
}

inline void cblas_Xsbmv1(
    const MatrixIndexT dim,
    const float *A,
    const float alpha,
    const float *x,
    const float beta,
    float *y) {
  assert(0 && "dummy cblas library has no implementations");
}

/// This is not really a wrapper for CBLAS as CBLAS does not have this; in future we could
/// extend this somehow.
inline void mul_elements(
    const MatrixIndexT dim,
    const double *a,
    double *b) { // does b *= a, elementwise.
  assert(0 && "dummy cblas library has no implementations");
}

inline void mul_elements(
    const MatrixIndexT dim,
    const float *a,
    float *b) { // does b *= a, elementwise.
  assert(0 && "dummy cblas library has no implementations");
}



// add clapack here
#if !defined(HAVE_ATLAS)
inline void clapack_Xtptri(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void clapack_Xtptri(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
// 
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols, 
                            float *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot, 
                            KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void clapack_Xgetrf2(KaldiBlasInt *num_rows, KaldiBlasInt *num_cols, 
                            double *Mdata, KaldiBlasInt *stride, KaldiBlasInt *pivot, 
                            KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}

// 
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, float *p_work, 
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void clapack_Xgetri2(KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           KaldiBlasInt *pivot, double *p_work, 
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
//
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, float *Mdata, KaldiBlasInt *stride,
                           float *sv, float *Vdata, KaldiBlasInt *vstride,
                           float *Udata, KaldiBlasInt *ustride, float *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void clapack_Xgesvd(char *v, char *u, KaldiBlasInt *num_cols,
                           KaldiBlasInt *num_rows, double *Mdata, KaldiBlasInt *stride,
                           double *sv, double *Vdata, KaldiBlasInt *vstride,
                           double *Udata, KaldiBlasInt *ustride, double *p_work,
                           KaldiBlasInt *l_work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
//
void inline clapack_Xsptri(KaldiBlasInt *num_rows, float *Mdata, 
                           KaldiBlasInt *ipiv, float *work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
void inline clapack_Xsptri(KaldiBlasInt *num_rows, double *Mdata, 
                           KaldiBlasInt *ipiv, double *work, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
//
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, float *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
void inline clapack_Xsptrf(KaldiBlasInt *num_rows, double *Mdata,
                           KaldiBlasInt *ipiv, KaldiBlasInt *result) {
  assert(0 && "dummy cblas library has no implementations");
}
#else
inline void clapack_Xgetrf(MatrixIndexT num_rows, MatrixIndexT num_cols,
                           float *Mdata, MatrixIndexT stride, 
                           int *pivot, int *result) {
  assert(0 && "dummy cblas library has no implementations");
}

inline void clapack_Xgetrf(MatrixIndexT num_rows, MatrixIndexT num_cols,
                           double *Mdata, MatrixIndexT stride, 
                           int *pivot, int *result) {
  assert(0 && "dummy cblas library has no implementations");
}
//
inline int clapack_Xtrtri(int num_rows, float *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}

inline int clapack_Xtrtri(int num_rows, double *Mdata, MatrixIndexT stride) {
  assert(0 && "dummy cblas library has no implementations");
}
//
inline void clapack_Xgetri(MatrixIndexT num_rows, float *Mdata, MatrixIndexT stride,
                      int *pivot, int *result) {
  assert(0 && "dummy cblas library has no implementations");
}
inline void clapack_Xgetri(MatrixIndexT num_rows, double *Mdata, MatrixIndexT stride,
                      int *pivot, int *result) {
  assert(0 && "dummy cblas library has no implementations");
}
#endif

}
// namespace kaldi

#endif
