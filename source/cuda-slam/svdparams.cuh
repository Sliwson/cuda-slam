#pragma once

#include "cudacommon.cuh"

struct CudaSvdParams
{
	CudaSvdParams(int beforeLength, int afterLength, int m, int n, bool useMatrixU = true, bool useMatrixV = true);
	void Free();

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
