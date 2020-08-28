#pragma once

#include "cuda.cuh"

struct CudaParallelSvdHelper
{
	CudaParallelSvdHelper(int batchSize, int m, int n, bool useMatrixU = true, bool useMatrixV = true)
		:batchSize(batchSize), m(m), n(n), useMatrixU(useMatrixU), useMatrixV(useMatrixV)
	{
		info.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			error = cudaMalloc(&(info[i]), sizeof(int));
			assert(error == cudaSuccess);
		}
		
		S.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			error = cudaMalloc(&(S[i]), n * n * sizeof(float));
			assert(error == cudaSuccess);
		}

		VT.resize(batchSize);
		if (useMatrixV)
		{
			for (int i = 0; i < batchSize; i++)
			{
				error = cudaMalloc(&(VT[i]), n * n * sizeof(float));
				assert(error == cudaSuccess);
			}
		}

		U.resize(batchSize);
		if (useMatrixU)
		{
			for (int i = 0; i < batchSize; i++)
			{
				error = cudaMalloc(&(U[i]), m * m * sizeof(float));
				assert(error == cudaSuccess);
			}
		}

		// Create handles and streams
		solverHandles.resize(batchSize);
		streams.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			cusolverStatus = cusolverDnCreate(&(solverHandles[i]));
			assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);

			error = cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking);
			assert(error == cudaSuccess);
			
			cusolverStatus = cusolverDnSetStream(solverHandles[i], streams[i]);
			assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);

		}

		// Allocate memory for SVD work
		work.resize(batchSize);
		workSize.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			cusolverStatus = cusolverDnSgesvd_bufferSize(solverHandles[i], m, n, &(workSize[i]));
			assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);

			error = cudaMalloc(&(work[i]), workSize[i] * sizeof(float));
			assert(error == cudaSuccess);
		}
	}

	void RunSVD(const thrust::host_vector<float*>& sourceMatrices)
	{
		for (int i = 0; i < batchSize; i++)
		{
			cusolverStatus = cusolverDnSgesvd(solverHandles[i], 'N', 'A', m, n, sourceMatrices[i], m, S[i], U[i], m, VT[i], n, work[i], workSize[i], nullptr, info[i]);
			assert(cusolverStatus == CUSOLVER_STATUS_SUCCESS);
		}
		error = cudaDeviceSynchronize();
		assert(error == cudaSuccess);

		//cusolverStatus_t status = cusolverDnSgesvdjBatched_bufferSize(solverHandles, CUSOLVER_EIG_MODE_VECTOR, m, n, sourceMatrices, m, S, U, m, V, n, &workSize, gesvdj_info, batchSize);
		//assert(CUSOLVER_STATUS_SUCCESS == status);

		//cudaError_t cudaStat = cudaMalloc(&work, workSize * sizeof(float));
		//assert(cudaSuccess == cudaStat);

		//cusolverDnSgesvd(solverHandles[0], 'N', 'A', m, n, params.workBefore, params.m, params.S, params.U, params.m, params.VT, params.n, params.work, params.workSize, nullptr, params.devInfo);

		//cusolverDnDgesvd_bufferSize(solverHandles[0], m, n, &workSize);
		//cusolverDnSgesvd()
	}

	thrust::host_vector<glm::mat3> GetHostMatricesVT()
	{
		thrust::host_vector<glm::mat3> result(batchSize);
		
		float* data = (float*)malloc(n * n * sizeof(float));
		for (int i = 0; i < batchSize; i++)
		{
			// Use V^T matrix instead of U as we pass transposed matrix to cusolver
			// A = U * S * V => A^T = V^T * S^T * U^T => U(A^T)  = V^T (more or less :) )
			error = cudaMemcpy(data, VT[i], 9 * sizeof(float), cudaMemcpyDeviceToHost);
			assert(error == cudaSuccess);

			printf("-----%d-----\n", i);
			for (int j = 0; j < 9; j++)
				printf("%f\t", data[j]);
			printf("\n");

			result[i] = CreateGlmMatrix(data);

			for (int j = 0; j < 9; j++)
				printf("%f\t", result[i][j%3][j/3]);
			printf("\n");
			printf("------------\n", i);
		}
		free(data);

		return result;
	}

	void FreeMemory()
	{
		for (int i = 0; i < batchSize; i++)
		{
			if(streams[i])
				cudaStreamDestroy(streams[i]);
			
			if(solverHandles[i])
				cusolverDnDestroy(solverHandles[i]);

			if (work[i])
				cudaFree(work[i]);

			if (S[i])
				cudaFree(S[i]);

			if (VT[i])
				cudaFree(VT[i]);

			if (U[i])
				cudaFree(U[i]);

			if (info[i])
				cudaFree(info[i]);
		}

		cudaDeviceReset();
	}

	int batchSize = 0;
	int m = 0;
	int n = 0;
	bool useMatrixU = true;
	bool useMatrixV = true;

	thrust::host_vector<cusolverDnHandle_t> solverHandles;
	thrust::host_vector<cudaStream_t> streams;

	thrust::host_vector<float*> S;
	thrust::host_vector<float*> VT;
	thrust::host_vector<float*> U;
	thrust::host_vector<int> workSize;
	thrust::host_vector<float*> work;
	thrust::host_vector<int*> info;

	cudaError_t error = cudaSuccess;
	cusolverStatus_t cusolverStatus = CUSOLVER_STATUS_SUCCESS;
};
