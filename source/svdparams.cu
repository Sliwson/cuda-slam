#include "svdparams.cuh"

using namespace CUDACommon;

CudaSvdParams::CudaSvdParams(int beforeLength, int afterLength, int m, int n, bool useMatrixU, bool useMatrixV)
	:m(m), n(n), useMatrixU(useMatrixU), useMatrixV(useMatrixV)
{
	checkCudaErrors(cudaMalloc((void**)&workBefore, beforeLength * n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&workAfter, afterLength * n * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&multiplyResult, n * m * sizeof(float)));
	cublasCreate(&multiplyHandle);

	checkCudaErrors(cudaMalloc((void**)&devInfo, sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&S, n * n * sizeof(float)));
	if (useMatrixV)
		checkCudaErrors(cudaMalloc((void**)&VT, n * n * sizeof(float)));
	if (useMatrixU)
		checkCudaErrors(cudaMalloc((void**)&U, m * m * sizeof(float)));
	cusolveSafeCall(cusolverDnCreate(&solverHandle));

	cusolveSafeCall(cusolverDnSgesvd_bufferSize(solverHandle, m, n, &workSize));

	checkCudaErrors(cudaMalloc((void**)&work, workSize * sizeof(float)));
}

void CudaSvdParams::Free()
{
	checkCudaErrors(cudaFree(workBefore));
	checkCudaErrors(cudaFree(workAfter));
	checkCudaErrors(cudaFree(multiplyResult));
	cublasDestroy(multiplyHandle);

	checkCudaErrors(cudaFree(work));
	checkCudaErrors(cudaFree(devInfo));
	checkCudaErrors(cudaFree(S));
	if (useMatrixV)
		checkCudaErrors(cudaFree(VT));
	if (useMatrixU)
		checkCudaErrors(cudaFree(U));
	cusolveSafeCall(cusolverDnDestroy(solverHandle));
}
