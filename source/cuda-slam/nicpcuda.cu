#include "nicpcuda.cuh"
#include "functors.cuh"
#include "nicputils.h"
#include "nicpslamargs.cuh"

using namespace CUDACommon;
using namespace Common;

namespace
{
	void PrepareMatricesForParallelSVD(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter, int batchSize, NonIterativeSLAMArgs& args)
	{
		int cloudSize = std::min(cloudBefore.size(), cloudAfter.size());

		for (int i = 0; i < batchSize; i++)
		{
			// Generate permutation
			std::vector<int> h_permutation = GetRandomPermutationVector(cloudSize);
			IndexIterator d_permutation(h_permutation.size());
			thrust::copy(h_permutation.begin(), h_permutation.end(), d_permutation.begin());
			ApplyPermutation(args.alignedCloudBefore, d_permutation, args.permutedCloudBefore);
			ApplyPermutation(args.alignedCloudAfter, d_permutation, args.permutedCloudAfter);

			// Create counting iterators
			auto beforeCountingBegin = thrust::make_counting_iterator<int>(0);
			auto beforeCountingEnd = thrust::make_counting_iterator<int>(args.permutedCloudBefore.size());
			auto afterCountingBegin = thrust::make_counting_iterator<int>(0);
			auto afterCountingEnd = thrust::make_counting_iterator<int>(args.permutedCloudAfter.size());

			// Create array for SVD
			const auto beforeZipBegin = thrust::make_zip_iterator(thrust::make_tuple(beforeCountingBegin, args.permutedCloudBefore.begin()));
			const auto beforeZipEnd = thrust::make_zip_iterator(thrust::make_tuple(beforeCountingEnd, args.permutedCloudBefore.end()));
			auto convertBefore = Functors::GlmToCuBlas(true, args.permutedCloudBefore.size(), args.preparedBeforeClouds[i]);
			thrust::for_each(thrust::device, beforeZipBegin, beforeZipEnd, convertBefore);
			const auto afterZipBegin = thrust::make_zip_iterator(thrust::make_tuple(afterCountingBegin, args.permutedCloudAfter.begin()));
			const auto afterZipEnd = thrust::make_zip_iterator(thrust::make_tuple(afterCountingEnd, args.permutedCloudAfter.end()));
			auto convertAfter = Functors::GlmToCuBlas(true, args.permutedCloudAfter.size(), args.preparedAfterClouds[i]);
			thrust::for_each(thrust::device, afterZipBegin, afterZipEnd, convertAfter);
		}
	}

	void GetSVDResultParallel(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter, int batchSize, NonIterativeSLAMArgs& args, thrust::host_vector<glm::mat3>& outputBefore, thrust::host_vector<glm::mat3>& outputAfter)
	{
		PrepareMatricesForParallelSVD(cloudBefore, cloudAfter, batchSize, args);

		// Run SVD for cloud before
		args.svdHelperBefore.RunSVD(args.preparedBeforeClouds, batchSize);
		outputBefore = args.svdHelperBefore.GetHostMatricesVT();

		args.svdHelperAfter.RunSVD(args.preparedAfterClouds, batchSize);
		outputAfter = args.svdHelperAfter.GetHostMatricesVT();
	}

	void GetSubcloud(const GpuCloud& cloud, int subcloudSize, GpuCloud& outputSubcloud)
	{
		if (subcloudSize >= cloud.size())
			outputSubcloud = cloud;

		outputSubcloud.resize(subcloudSize);

		std::vector<int> h_indices = GetRandomPermutationVector(cloud.size());
		h_indices.resize(subcloudSize);
		thrust::device_vector<int> d_indices(h_indices);

		auto permutationIterBegin = thrust::make_permutation_iterator(cloud.begin(), d_indices.begin());
		auto permutationIterEnd = thrust::make_permutation_iterator(cloud.end(), d_indices.end());
		thrust::copy(permutationIterBegin, permutationIterEnd, outputSubcloud.begin());
	}
}

std::pair<glm::mat3, glm::vec3> CudaNonIterative(const GpuCloud& before, const GpuCloud& after, int* repetitions, float* error, float eps, int maxRepetitions, int batchSize, ApproximationType approximationType, const int subcloudSize)
{
	// Set the number of results to store - 1 for Full, 5 for Hybrid, unused for None
	auto resultsNumber = approximationType == ApproximationType::Full ? 1 : 5;
	
	// Prepare stuctures for storing transformation results for subsequent executions
	std::pair<glm::mat3, glm::vec3> bestTransformation;
	std::pair<glm::mat3, glm::vec3> currentTransformation;
	thrust::host_vector<glm::mat3> matricesBefore(batchSize);
	thrust::host_vector<glm::mat3> matricesAfter(batchSize);
	std::vector<NonIterativeSlamResult> bestResults(resultsNumber);

	// Split number of repetitions to batches
	auto batchesCount = maxRepetitions / batchSize;
	auto lastBatchSize = maxRepetitions % batchSize;
	auto threadsToRun = batchSize;

	// Prepare helper structures
	auto minError = *error = std::numeric_limits<float>::max();
	GpuCloud subcloud(subcloudSize);
	GpuCloud transformedSubcloud(subcloudSize);
	GetSubcloud(before, subcloudSize, subcloud);
	thrust::device_vector<int> permutedIndices(subcloudSize);
	thrust::device_vector<int> nonPermutedIndices(before.size());
	thrust::counting_iterator<int> helperIterator(0);
	thrust::copy(helperIterator, helperIterator + before.size(), nonPermutedIndices.begin());

	NonIterativeSLAMArgs args(batchSize, before, after);

	auto centroidBefore = CalculateCentroid(before);
	auto centroidAfter = CalculateCentroid(after);

	*repetitions = 0;

	// Run actual SLAM in batches
	for (int i = 0; i <= batchesCount; i++)
	{
		if (i == batchesCount)
		{
			if (lastBatchSize != 0)
				threadsToRun = lastBatchSize;
			else
				break;
		}

		GetSVDResultParallel(before, after, threadsToRun, args, matricesBefore, matricesAfter);
		*repetitions += threadsToRun;

		for (int j = 0; j < threadsToRun; j++)
		{
			glm::mat3 rotationMatrix = matricesAfter[j] * glm::transpose(matricesBefore[j]);
			glm::vec3 translationVector = centroidAfter - (rotationMatrix * centroidBefore);
			currentTransformation = std::make_pair(rotationMatrix, translationVector);

			// If using approximation, get error without finding correspondences - quick and efficient for poorly permuted clouds
			// Do find correspondences if using approximationType == None
			if (approximationType == ApproximationType::None)
			{
				TransformCloud(subcloud, transformedSubcloud, ConvertToTransformationMatrix(currentTransformation.first, currentTransformation.second));
				GetCorrespondingPoints(permutedIndices, transformedSubcloud, after);
				*error = GetMeanSquaredError(permutedIndices, transformedSubcloud, after);

				if (*error < minError)
				{
					minError = *error;
					bestTransformation = currentTransformation;

					if (minError <= eps)
					{
						printf("Error: %f\n", minError);
						args.Free();
						return currentTransformation;
					}
				}
			}
			else
			{
				*error = GetMeanSquaredError(nonPermutedIndices, before, after);

				NonIterativeSlamResult transformationResult(rotationMatrix, translationVector, *error);
				StoreResultIfOptimal(bestResults, transformationResult, resultsNumber);
			}
		}
	}

	// If using hybrid approximation, select best result
	// If using full approximation, calculate exact error for the best result
	if (approximationType != ApproximationType::None)
	{
		minError = std::numeric_limits<float>::max();
		for (int i = 0; i < bestResults.size(); i++)
		{
			TransformCloud(subcloud, transformedSubcloud, bestResults[i].getTransformationMatrix());
			GetCorrespondingPoints(permutedIndices, transformedSubcloud, after);
			*error = GetMeanSquaredError(permutedIndices, transformedSubcloud, after);

			if (*error < minError)
			{
				minError = *error;
				bestTransformation = bestResults[i].getTransformation();

				if (minError <= eps)
				{
					args.Free();
					return bestTransformation;
				}
			}
		}
	}

	*error = minError;
	args.Free();
	return bestTransformation;
}

std::pair<glm::mat3, glm::vec3> GetCudaNicpTransformationMatrix(
	const std::vector<Point_f>& before,
	const std::vector<Point_f>& after,
	float eps,
	int maxRepetitions,
	int batchSize,
	Common::ApproximationType approximationType,
	const int subcloudSize,
	int* repetitions,
	float* error)
{
	GpuCloud gpuBefore(before.size());
	GpuCloud gpuAfter(after.size());

	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(gpuBefore.data()), before.data(), before.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(gpuAfter.data()), after.data(), after.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

	const auto result = CudaNonIterative(gpuBefore, gpuAfter, repetitions, error, eps, maxRepetitions, batchSize, approximationType, subcloudSize);

	GpuCloud transformedCloud(gpuBefore.size());
	TransformCloud(gpuBefore, transformedCloud, ConvertToTransformationMatrix(result.first, result.second));
	thrust::device_vector<int> indices(gpuBefore.size());
	GetCorrespondingPoints(indices, transformedCloud, gpuAfter);
	*error = GetMeanSquaredError(indices, transformedCloud, gpuAfter);

	return result;
}

