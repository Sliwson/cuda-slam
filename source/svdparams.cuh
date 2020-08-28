#pragma once

#include "cuda.cuh"

struct CudaSvdParams
{
	CudaSvdParams(int beforeLength, int afterLength,)
	{
		cudaMalloc(&workBefore, beforeLength * 3 * sizeof(float));
		cudaMalloc(&workAfter, afterLength * 3 * sizeof(float));
		cudaMalloc(&multiplyResult, 3 * 3 * sizeof(float));
		cublasCreate(&multiplyHandle);

		cudaMalloc(&devInfo, sizeof(int));
		cudaMalloc(&S, 9 * sizeof(float));
		cudaMalloc(&VT, 9 * sizeof(float));
		cudaMalloc(&U, 9 * sizeof(float));
		cusolverDnCreate(&solverHandle);
		
		cusolverDnDgesvd_bufferSize(solverHandle, 3, 3, &workSize);
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
	cusolverDnHandle_t solverHandle = nullptr;
};
