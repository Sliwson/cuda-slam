#pragma once

#include "cuda.cuh"
#include "cudaprobabilities.h"

struct CudaMStepParams
{
	CudaMStepParams(const int& beforeLength, const int& afterLength, const CUDAProbabilities::Probabilities& probabilities)
	{
		cudaMalloc(&beforeT, beforeLength * 3 * sizeof(float));
		cudaMalloc(&afterT, afterLength * 3 * sizeof(float));
		cudaMalloc(&centerBefore, 3 * 3 * sizeof(float));
		cudaMalloc(&centerAfter, 3 * 3 * sizeof(float));
		cudaMalloc(&centerBefore, 3 * 3 * sizeof(float));
		cudaMalloc(&centerBefore, 3 * 3 * sizeof(float));



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

	float* beforeT = nullptr;
	float* afterT = nullptr;
	float* centerBefore = nullptr;
	float* centerAfter = nullptr;
	float* AMatrix = nullptr;
	float* p1 = nullptr;
	float* pt1 = nullptr;
	float* px = nullptr;

	cublasHandle_t multiplyHandle = nullptr;

	//for svd
	float* S = nullptr;
	float* VT = nullptr;
	float* U = nullptr;
	float* work = nullptr;
	int* devInfo = nullptr;
	int workSize = 0;
	cusolverDnHandle_t solverHandle = nullptr;
};
