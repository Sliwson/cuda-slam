#include "parallelsvdhelper.cuh"

CudaParallelSvdHelper::CudaParallelSvdHelper(int batchSize, int m, int n, bool useMatrixU, bool useMatrixV)
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

void CudaParallelSvdHelper::RunSVD(const thrust::host_vector<float*>& sourceMatrices)
{
	const auto thread_work = [&](int index) {
		auto status = cusolverDnSgesvd(solverHandles[index], 'N', 'A', m, n, sourceMatrices[index], m, S[index], U[index], m, VT[index], n, work[index], workSize[index], nullptr, info[index]);
		assert(status == CUSOLVER_STATUS_SUCCESS);
	};

	std::vector<std::thread> workerThreads(batchSize);

	// SVD needs to be launched from separated threads to take full advantage of CUDA streams
	for (int j = 0; j < batchSize; j++)
		workerThreads[j] = std::thread(thread_work, j);

	// Wait for threads to finish
	for (int j = 0; j < batchSize; j++)
		workerThreads[j].join();

	error = cudaDeviceSynchronize();
	assert(error == cudaSuccess);
}

thrust::host_vector<glm::mat3> CudaParallelSvdHelper::GetHostMatricesVT()
{
	thrust::host_vector<glm::mat3> result(batchSize);

	float* data = (float*)malloc(n * n * sizeof(float));
	for (int i = 0; i < batchSize; i++)
	{
		// Use V^T matrix instead of U as we pass transposed matrix to cusolver
		// A = U * S * V => A^T = V^T * S^T * U^T => U(A^T)  = V^T (more or less :) )
		error = cudaMemcpy(data, VT[i], 9 * sizeof(float), cudaMemcpyDeviceToHost);
		assert(error == cudaSuccess);

		result[i] = CUDACommon::CreateGlmMatrix(data);
	}
	free(data);

	return result;
}

void CudaParallelSvdHelper::FreeMemory()
{
	for (int i = 0; i < batchSize; i++)
	{
		if (streams[i])
			cudaStreamDestroy(streams[i]);

		if (solverHandles[i])
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
}