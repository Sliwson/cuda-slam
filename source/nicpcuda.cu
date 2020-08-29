#include "icpcuda.cuh"
#include "functors.cuh"
#include "parallelsvdhelper.cuh"

using namespace CUDACommon;

namespace
{
	void PrepareMatricesForParallelSVD(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter, int batchSize, thrust::host_vector<float*>& outputBefore, thrust::host_vector<float*>& outputAfter)
	{
		int cloudSize = std::min(cloudBefore.size(), cloudAfter.size());

		outputBefore.resize(batchSize);
		outputAfter.resize(batchSize);
		for (int i = 0; i < batchSize; i++)
		{
			cudaMalloc(&(outputBefore[i]), 3 * cloudBefore.size() * sizeof(float));
			cudaMalloc(&(outputAfter[i]), 3 * cloudAfter.size() * sizeof(float));
		}

		GpuCloud alignBefore(cloudBefore.size());
		GpuCloud alignAfter(cloudAfter.size());

		// Align array
		GetAlignedCloud(cloudBefore, alignBefore);
		GetAlignedCloud(cloudAfter, alignAfter);

		for (int i = 0; i < batchSize; i++)
		{
			// Generate permutation
			std::vector<int> h_permutation = GetRandomPermutationVector(cloudSize);
			IndexIterator d_permutation(h_permutation.size());
			thrust::copy(h_permutation.begin(), h_permutation.end(), d_permutation.begin());
			auto permutedBefore = ApplyPermutation(alignBefore, d_permutation);
			auto permutedAfter = ApplyPermutation(alignAfter, d_permutation);

			// Create counting iterators
			auto beforeCountingBegin = thrust::make_counting_iterator<int>(0);
			auto beforeCountingEnd = thrust::make_counting_iterator<int>(permutedBefore.size());
			auto afterCountingBegin = thrust::make_counting_iterator<int>(0);
			auto afterCountingEnd = thrust::make_counting_iterator<int>(permutedAfter.size());

			// Create array for SVD
			const auto beforeZipBegin = thrust::make_zip_iterator(thrust::make_tuple(beforeCountingBegin, permutedBefore.begin()));
			const auto beforeZipEnd = thrust::make_zip_iterator(thrust::make_tuple(beforeCountingEnd, permutedBefore.end()));
			auto convertBefore = Functors::GlmToCuBlas(true, permutedBefore.size(), outputBefore[i]);
			thrust::for_each(thrust::device, beforeZipBegin, beforeZipEnd, convertBefore);
			const auto afterZipBegin = thrust::make_zip_iterator(thrust::make_tuple(afterCountingBegin, permutedAfter.begin()));
			const auto afterZipEnd = thrust::make_zip_iterator(thrust::make_tuple(afterCountingEnd, permutedAfter.end()));
			auto convertAfter = Functors::GlmToCuBlas(true, permutedAfter.size(), outputAfter[i]);
			thrust::for_each(thrust::device, afterZipBegin, afterZipEnd, convertAfter);
		}
	}

	void GetSVDResultParallel(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter, int batchSize, thrust::host_vector<glm::mat3>& outputBefore, thrust::host_vector<glm::mat3>& outputAfter)
	{
		thrust::host_vector<float*> preparedBefore;
		thrust::host_vector<float*> preparedAfter;

		PrepareMatricesForParallelSVD(cloudBefore, cloudAfter, batchSize, preparedBefore, preparedAfter);

		// Run SVD for cloud before
		CudaParallelSvdHelper svdBefore(batchSize, cloudBefore.size(), 3, false);
		svdBefore.RunSVD(preparedBefore);
		outputBefore = svdBefore.GetHostMatricesVT();
		svdBefore.FreeMemory();

		// Run SVD for cloud after
		CudaParallelSvdHelper svdAfter(batchSize, cloudAfter.size(), 3, false);
		svdAfter.RunSVD(preparedAfter);
		outputAfter = svdAfter.GetHostMatricesVT();
		svdAfter.FreeMemory();

		for (int i = 0; i < batchSize; i++)
		{
			if (preparedBefore[i])
				cudaFree(preparedBefore[i]);

			if (preparedAfter[i])
				cudaFree(preparedAfter[i]);
		}
	}

	GpuCloud GetSubcloud(const GpuCloud& cloud, int subcloudSize)
	{
		if (subcloudSize >= cloud.size())
			return cloud;
		std::vector<int> subcloudIndices = GetRandomPermutationVector(cloud.size());
		subcloudIndices.resize(subcloudSize);
		thrust::device_vector<int> indices(subcloudIndices);

		GpuCloud subcloud(subcloudIndices.size());
		const auto getSubcloudFunctor = Functors::Permutation(cloud);
		thrust::transform(thrust::device, indices.begin(), indices.end(), subcloud.begin(), getSubcloudFunctor);

		return subcloud;
	}

	glm::mat4 CudaNonIterative(const GpuCloud& before, const GpuCloud& after, int* repetitions, float* error, float eps, int maxRepetitions, int batchSize, const int subcloudSize)
	{
		glm::mat4 transformResult(1.0f);
		*error = std::numeric_limits<float>::max();

		thrust::host_vector<glm::mat3> matricesBefore(batchSize);
		thrust::host_vector<glm::mat3> matricesAfter(batchSize);

		const auto subcloud = GetSubcloud(before, subcloudSize);
		auto batchesCount = maxRepetitions / batchSize;
		auto lastBatchSize = maxRepetitions % batchSize;
		auto threadsToRun = batchSize;

		auto centroidBefore = CalculateCentroid(before);
		auto centroidAfter = CalculateCentroid(after);

		for (int i = 0; i <= batchesCount; i++)
		{
			if (i == batchesCount)
			{
				if (lastBatchSize != 0)
					threadsToRun = lastBatchSize;
				else
					break;
			}

			GetSVDResultParallel(before, after, threadsToRun, matricesBefore, matricesAfter);
			*repetitions += threadsToRun;

			for (int j = 0; j < threadsToRun; j++)
			{
				auto transformationMatrix = glm::mat4(1.0f);
				auto rotationMatrix = matricesAfter[j] * glm::transpose(matricesBefore[j]);
				auto translationVector = centroidAfter - (rotationMatrix * centroidBefore);

				for (int x = 0; x < 3; x++)
					for (int y = 0; y < 3; y++)
						transformationMatrix[x][y] = rotationMatrix[x][y];

				transformationMatrix[3][0] = translationVector.x;
				transformationMatrix[3][1] = translationVector.y;
				transformationMatrix[3][2] = translationVector.z;
				transformationMatrix[3][3] = 1.0f;

				GpuCloud workingSubcloud(subcloud.size());
				thrust::device_vector<int> indices(workingSubcloud.size());
				TransformCloud(subcloud, workingSubcloud, transformationMatrix);
				GetCorrespondingPoints(indices, workingSubcloud, after);
				auto currentError = GetMeanSquaredError(indices, workingSubcloud, after);
				printf("Current error: %f\n", currentError);

				// Process the results
				if (currentError <= eps)
				{
					*error = currentError;
					return transformationMatrix;
				}

				if (currentError < *error)
				{
					*error = currentError;
					transformResult = transformationMatrix;
				}
			}
		}

		return transformResult;
	}
}

void NonIterativeCudaTest()
{
	/****************************/
	//TESTS
	/****************************/
	//MultiplicationTest();

	/****************************/
	//ALGORITHM
	/****************************/
	const auto testCloud = LoadCloud("data/bunny.obj");
	const auto testCorrupted = LoadCloud("data/bunny.obj");
	int repetitions;
	float error;
	const int maxRepetitions = 20;
	const int subcloudSize = 1000;
	const float eps = 1e-5;
	const int cpuThreadsCount = (int)std::thread::hardware_concurrency();
	//testCloud.resize(10000);
	//testCorrupted.resize(10000);

	const auto hostBefore = CommonToThrustVector(testCloud);
	const auto hostAfter = CommonToThrustVector(testCorrupted);

	GpuCloud deviceCloudBefore = hostBefore;
	GpuCloud deviceCloudAfter = hostAfter;

	GpuCloud calculatedCloud(hostAfter.size());

	const auto scaleInput = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudBefore.begin(), scaleInput);
	const auto scaleInputCorrupted = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudAfter.begin(), deviceCloudAfter.end(), deviceCloudAfter.begin(), scaleInputCorrupted);

	const auto sampleTransform = glm::rotate(glm::translate(glm::mat4(1), { 0.05f, 0.05f, 0.05f }), glm::radians(5.f), { 0.5f, 0.5f, 0.5f });
	// TODO: Remove this debug print for sample transformation
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("%f\t", sampleTransform[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	TransformCloud(deviceCloudAfter, deviceCloudAfter, sampleTransform);

	auto start = std::chrono::high_resolution_clock::now();
	const auto result = CudaNonIterative(deviceCloudBefore, deviceCloudAfter, &repetitions, &error, eps, maxRepetitions, cpuThreadsCount, subcloudSize);
	auto stop = std::chrono::high_resolution_clock::now();
	printf("Size: %d points, duration: %dms\n", testCloud.size(), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start));

	TransformCloud(deviceCloudBefore, calculatedCloud, result);

	Common::Renderer renderer(
		Common::ShaderType::SimpleModel,
		ThrustToCommonVector(deviceCloudBefore), //red
		ThrustToCommonVector(deviceCloudAfter), //green
		ThrustToCommonVector(calculatedCloud), //yellow
		std::vector<Point_f>(1));

	renderer.Show();
}
