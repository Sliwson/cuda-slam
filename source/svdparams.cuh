#pragma once

#include "cuda.cuh"

struct CudaSvdParams
{
	CudaSvdParams(int beforeLength, int afterLength, int m, int n)
		:m(m), n(n)
	{
		cudaMalloc(&workBefore, beforeLength * n * sizeof(float));
		cudaMalloc(&workAfter, afterLength * n * sizeof(float));
		cudaMalloc(&multiplyResult, n * m * sizeof(float));
		cublasCreate(&multiplyHandle);

		cudaMalloc(&devInfo, sizeof(int));
		cudaMalloc(&S, n * n * sizeof(float));
		cudaMalloc(&VT, n * n * sizeof(float));
		cudaMalloc(&U, m * m * sizeof(float));
		cusolverDnCreate(&solverHandle);

		cusolverDnDgesvd_bufferSize(solverHandle, m, n, &workSize);
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
		cudaFree(VT);
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
	cusolverDnHandle_t solverHandle = nullptr;
};
