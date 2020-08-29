#include "svdparams.cuh"

CudaSvdParams::CudaSvdParams(int beforeLength, int afterLength, int m, int n, bool useMatrixU, bool useMatrixV)
	:m(m), n(n), useMatrixU(useMatrixU), useMatrixV(useMatrixV)
{
	cudaMalloc(&workBefore, beforeLength * n * sizeof(float));
	cudaMalloc(&workAfter, afterLength * n * sizeof(float));
	cudaMalloc(&multiplyResult, n * m * sizeof(float));
	cublasCreate(&multiplyHandle);

	cudaMalloc(&devInfo, sizeof(int));
	cudaMalloc(&S, n * n * sizeof(float));
	if (useMatrixV)
		cudaMalloc(&VT, n * n * sizeof(float));
	if (useMatrixU)
		cudaMalloc(&U, m * m * sizeof(float));
	cusolverDnCreate(&solverHandle);

	cusolverDnSgesvd_bufferSize(solverHandle, m, n, &workSize);

	cudaMalloc(&work, workSize * sizeof(float));
}

void CudaSvdParams::Free()
{
	cudaFree(workBefore);
	cudaFree(workAfter);
	cudaFree(multiplyResult);
	cublasDestroy(multiplyHandle);

	cudaFree(work);
	cudaFree(devInfo);
	cudaFree(S);
	if (useMatrixV)
		cudaFree(VT);
	if (useMatrixU)
		cudaFree(U);
	cusolverDnDestroy(solverHandle);
}
