#include "cudacommon.cuh"
#include "parallelsvdhelper.cuh"

using namespace CUDACommon;

struct NonIterativeSLAMArgs
{
public:
	NonIterativeSLAMArgs(int batchSize, const GpuCloud& cloudBefore, const GpuCloud& cloudAfter)
		:batchSize(batchSize), 
		svdHelperBefore(batchSize, cloudBefore.size(), 3, false), 
		svdHelperAfter(batchSize, cloudAfter.size(), 3, false), 
		alignedCloudBefore(cloudBefore.size()), 
		alignedCloudAfter(cloudAfter.size()),
		permutedCloudBefore(cloudBefore.size()),
		permutedCloudAfter(cloudAfter.size())
	{
		preparedBeforeClouds.resize(batchSize);
		preparedAfterClouds.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			cudaMalloc(&(preparedBeforeClouds[i]), 3 * cloudBefore.size() * sizeof(float));
			cudaMalloc(&(preparedAfterClouds[i]), 3 * cloudAfter.size() * sizeof(float));
		}

		GetAlignedCloud(cloudBefore, alignedCloudBefore);
		GetAlignedCloud(cloudAfter, alignedCloudAfter);
	}

	void Free()
	{
		for (int i = 0; i < batchSize; i++)
		{
			cudaFree(preparedBeforeClouds[i]);
			cudaFree(preparedAfterClouds[i]);
		}

		svdHelperBefore.FreeMemory();
		svdHelperAfter.FreeMemory();
	}

public:
	int batchSize;

	thrust::host_vector<float*> preparedBeforeClouds;
	thrust::host_vector<float*> preparedAfterClouds;

	CudaParallelSvdHelper svdHelperBefore;
	CudaParallelSvdHelper svdHelperAfter;

	GpuCloud alignedCloudBefore;
	GpuCloud alignedCloudAfter;
	GpuCloud permutedCloudBefore;
	GpuCloud permutedCloudAfter;
};