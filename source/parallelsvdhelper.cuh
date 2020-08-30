#pragma once

#include "cuda.cuh"

struct CudaParallelSvdHelper
{
	CudaParallelSvdHelper(int batchSize, int m, int n, bool useMatrixU = true, bool useMatrixV = true);
	void RunSVD(const thrust::host_vector<float*>& sourceMatrices, int threadsToRun = -1);
	thrust::host_vector<glm::mat3> GetHostMatricesVT();
	void FreeMemory();

	int batchSize = 0;
	int threadsNumber = 0;
	int m = 0;
	int n = 0;
	bool useMatrixU = true;
	bool useMatrixV = true;
	float* dataMatrixVT = nullptr;

	thrust::host_vector<cusolverDnHandle_t> solverHandles;
	thrust::host_vector<cudaStream_t> streams;

	thrust::host_vector<float*> S;
	thrust::host_vector<float*> VT;
	thrust::host_vector<float*> U;
	thrust::host_vector<int> workSize;
	thrust::host_vector<float*> work;
	thrust::host_vector<int*> info;

	cusolverStatus_t cusolverStatus = CUSOLVER_STATUS_SUCCESS;
};
