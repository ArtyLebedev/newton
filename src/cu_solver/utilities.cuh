#ifndef CU_SOLVER_UTILITIES_CUH_
#define CU_SOLVER_UTILITIES_CUH_
#include "cuda.h"
namespace newtone {
extern "C" int iDivUp(int, int);
extern "C" void gpuErrchk(cudaError_t);
extern "C" void cusolveSafeCall(cusolverStatus_t);
extern "C" void cublasSafeCall(cublasStatus_t);
}

#endif /* CU_SOLVER_UTILITIES_CUH_ */
