#pragma once

#include "cuda.cuh"

struct CudaSvdParams
{
	CudaSvdParams(int beforeLength, int afterLength, int m, int n, bool useMatrixU = true, bool useMatrixV = true)
		:m(m), n(n), useMatrixU(useMatrixU), useMatrixV(useMatrixV)
	{
		cudaMalloc(&workBefore, beforeLength * n * sizeof(float));
		cudaMalloc(&workAfter, afterLength * n * sizeof(float));
		cudaMalloc(&multiplyResult, n * m * sizeof(float));
		cublasCreate(&multiplyHandle);

		cudaMalloc(&devInfo, sizeof(int));
		cudaMalloc(&S, n * n * sizeof(float));
		if(useMatrixV)
			cudaMalloc(&VT, n * n * sizeof(float));
		if(useMatrixU)
			cudaMalloc(&U, m * m * sizeof(float));
		cusolverDnCreate(&solverHandle);

		cusolverDnSgesvd_bufferSize(solverHandle, m, n, &workSize);

		cudaMalloc(&work, workSize * sizeof(float));
	}

	void Free()
	{
		cudaFree(workBefore);
		cudaFree(workAfter);
		cudaFree(multiplyResult);
		cublasDestroy(multiplyHandle);
		
		cudaFree(work);
		cudaFree(devInfo);
		cudaFree(S);
		if(useMatrixV)
			cudaFree(VT);
		if(useMatrixU)
			cudaFree(U);
		cusolverDnDestroy(solverHandle);
	}

	//for multiplication 
	float* workBefore = nullptr;
	float* workAfter = nullptr;
	float* multiplyResult = nullptr;
	cublasHandle_t multiplyHandle;

	//for svd
	float* S = nullptr;
	float* VT = nullptr;
	float* U = nullptr;
	float* work = nullptr;
	int* devInfo = nullptr;
	int workSize = 0;
	int m = 0;
	int n = 0;
	bool useMatrixU = true;
	bool useMatrixV = true;
	cusolverDnHandle_t solverHandle = nullptr;
};
