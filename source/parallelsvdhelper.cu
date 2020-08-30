#include "parallelsvdhelper.cuh"

using namespace CUDACommon;

CudaParallelSvdHelper::CudaParallelSvdHelper(int batchSize, int m, int n, bool useMatrixU, bool useMatrixV)
	:batchSize(batchSize), m(m), n(n), useMatrixU(useMatrixU), useMatrixV(useMatrixV)
{
	info.resize(batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		checkCudaErrors(cudaMalloc((void**)&(info[i]), sizeof(int)));
	}

	S.resize(batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		checkCudaErrors(cudaMalloc((void**)&(S[i]), n * n * sizeof(float)));
	}

	VT.resize(batchSize);
	if (useMatrixV)
	{
		for (int i = 0; i < batchSize; i++)
		{
			checkCudaErrors(cudaMalloc((void**)&(VT[i]), n * n * sizeof(float)));
		}
	}

	U.resize(batchSize);
	if (useMatrixU)
	{
		for (int i = 0; i < batchSize; i++)
		{
			checkCudaErrors(cudaMalloc((void**)&(U[i]), m * m * sizeof(float)));
		}
	}

	// Create handles and streams
	solverHandles.resize(batchSize);
	streams.resize(batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		cusolveSafeCall(cusolverDnCreate(&(solverHandles[i])));
		checkCudaErrors(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
		cusolveSafeCall(cusolverDnSetStream(solverHandles[i], streams[i]));
	}

	// Allocate memory for SVD work
	work.resize(batchSize);
	workSize.resize(batchSize);
	for (int i = 0; i < batchSize; i++)
	{
		cusolveSafeCall(cusolverDnSgesvd_bufferSize(solverHandles[i], m, n, &(workSize[i])));
		checkCudaErrors(cudaMalloc((void**)&(work[i]), workSize[i] * sizeof(float)));
	}

	dataMatrixVT = (float*)malloc(n * n * sizeof(float));
}

void CudaParallelSvdHelper::RunSVD(const thrust::host_vector<float*>& sourceMatrices, int threadsToRun)
{
	threadsNumber = threadsToRun == -1 ? batchSize : threadsToRun;

	const auto thread_work = [&](int index) {
		cusolveSafeCall(cusolverDnSgesvd(solverHandles[index], 'N', 'A', m, n, sourceMatrices[index], m, S[index], U[index], m, VT[index], n, work[index], workSize[index], nullptr, info[index]));
	};

	std::vector<std::thread> workerThreads(threadsNumber);

	// SVD needs to be launched from separated threads to take full advantage of CUDA streams
	for (int j = 0; j < threadsNumber; j++)
		workerThreads[j] = std::thread(thread_work, j);

	// Wait for threads to finish
	for (int j = 0; j < threadsNumber; j++)
		workerThreads[j].join();

	checkCudaErrors(cudaDeviceSynchronize());
}

thrust::host_vector<glm::mat3> CudaParallelSvdHelper::GetHostMatricesVT()
{
	thrust::host_vector<glm::mat3> result(threadsNumber);

	for (int i = 0; i < threadsNumber; i++)
	{
		// Use V^T matrix instead of U as we pass transposed matrix to cusolver
		// A = U * S * V => A^T = V^T * S^T * U^T => U(A^T)  = V^T (more or less :) )
		checkCudaErrors(cudaMemcpy(dataMatrixVT, VT[i], 9 * sizeof(float), cudaMemcpyDeviceToHost));
		result[i] = CUDACommon::CreateGlmMatrix(dataMatrixVT);
	}

	return result;
}

void CudaParallelSvdHelper::FreeMemory()
{
	for (int i = 0; i < batchSize; i++)
	{
		if (streams[i])
			checkCudaErrors(cudaStreamDestroy(streams[i]));

		if (solverHandles[i])
			cusolveSafeCall(cusolverDnDestroy(solverHandles[i]));

		if (work[i])
			checkCudaErrors(cudaFree(work[i]));

		if (S[i])
			checkCudaErrors(cudaFree(S[i]));

		if (VT[i])
			checkCudaErrors(cudaFree(VT[i]));

		if (U[i])
			checkCudaErrors(cudaFree(U[i]));

		if (info[i])
			checkCudaErrors(cudaFree(info[i]));
	}

	free(dataMatrixVT);
}