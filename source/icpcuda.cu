#include "icpcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"

using namespace Common;
using namespace CUDACommon;

namespace
{
	__global__ void FindCorrespondences(int* result, const glm::vec3* before, const glm::vec3* after, int beforeSize, int afterSize)
	{
		int targetIdx = blockDim.x * blockIdx.x + threadIdx.x;
		if (targetIdx < beforeSize)
		{
			const glm::vec3 vector = before[targetIdx];
			int nearestIdx = 0;
			float smallestError = GetDistanceSquared(vector, after[0]);
			for (int i = 1; i < afterSize; i++)
			{
				const auto dist = GetDistanceSquared(vector, after[i]);
				if (dist < smallestError)
				{
					smallestError = dist;
					nearestIdx = i;
				}
			}

			result[targetIdx] = nearestIdx;
		}
	}

	void GetCorrespondingPoints(thrust::device_vector<int>& indices, const Cloud& before, const Cloud& after)
	{
#ifdef USE_CORRESPONDENCES_KERNEL
		int* dIndices = thrust::raw_pointer_cast(indices.data());
		const glm::vec3* dBefore = thrust::raw_pointer_cast(before.data());
		const glm::vec3* dAfter = thrust::raw_pointer_cast(after.data());
		int beforeSize = before.size();
		int afterSize = after.size();

		constexpr int threadsPerBlock = 256;
		const int blocksPerGrid = (beforeSize + threadsPerBlock - 1) / threadsPerBlock;
		FindCorrespondences << <blocksPerGrid, threadsPerBlock >> > (dIndices, dBefore, dAfter, beforeSize, afterSize);
		cudaDeviceSynchronize();
#else
		const auto nearestFunctor = Functors::FindNearestIndex(after);
		thrust::transform(thrust::device, before.begin(), before.end(), indices.begin(), nearestFunctor);
#endif
	}

	glm::mat4 CudaICP(const Cloud& before, const Cloud& after)
	{
		const int maxIterations = 60;
		const float TEST_EPS = 1e-5;
		float previousError = std::numeric_limits<float>::max();

		int iterations = 0;
		glm::mat4 transformationMatrix(1.0f);
		glm::mat4 previousTransformationMatrix = transformationMatrix;

		//do not change before vector - copy it for calculations
		const int size = std::max(before.size(), after.size());
		Cloud workingBefore(size);
		Cloud alignBefore(size);
		Cloud alignAfter(size);
		thrust::device_vector<int> indices(before.size());
		thrust::copy(thrust::device, before.begin(), before.end(), workingBefore.begin());

		//allocate memory for cuBLAS
		CudaSvdParams params(size, size);

		while (iterations < maxIterations)
		{
			GetCorrespondingPoints(indices, workingBefore, after);

			transformationMatrix = LeastSquaresSVD(indices, workingBefore, after, alignBefore, alignAfter, params) * transformationMatrix;

			TransformCloud(before, workingBefore, transformationMatrix);
			float error = GetMeanSquaredError(indices, workingBefore, after);
			printf("Iteration: %d, error: %f\n", iterations, error);
			if (error < TEST_EPS)
				break;

			if (error > previousError)
			{
				printf("Error has increased, aborting\n");
				transformationMatrix = previousTransformationMatrix;
				break;
			}

			previousTransformationMatrix = transformationMatrix;
			previousError = error;
			iterations++;
		}

		params.Free();
		return transformationMatrix;
	}

	void CorrespondencesTest()
	{
		const int size = 100;
		thrust::device_vector<glm::vec3> input(size);
		thrust::device_vector<glm::vec3> output(size);
		thrust::device_vector<int> result(size);

		for (int i = 0; i < size; i++)
		{
			const auto vec = glm::vec3(i);
			input[i] = vec;
			output[size - i - 1] = vec;
		}

		GetCorrespondingPoints(result, input, output);
		thrust::host_vector<int> copy = result;
		bool ok = true;
		int hostArray[size];
		for (int i = 0; i < size; i++)
		{
			hostArray[i] = copy[i];
			if (copy[i] != size - i - 1)
				ok = false;
		}

		printf("Correspondence test [%s]\n", ok ? "OK" : "FAILED");
	}

	void MultiplicationTest()
	{
		const int size = 100;

		float ones[3 * size];
		for (int i = 0; i < 3 * size; i++)
			ones[i] = 1.f;

		CudaSvdParams params(size, size);
		cudaMemcpy(params.workBefore, ones, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(params.workAfter, ones, 3 * size * sizeof(float), cudaMemcpyHostToDevice);

		CuBlasMultiply(params.workBefore, params.workAfter, params.multiplyResult, size, params);

		float result[9];
		cudaMemcpy(result, params.multiplyResult, 9 * sizeof(float), cudaMemcpyDeviceToHost);

		bool ok = true;
		for (int i = 0; i < 9; i++)
			if (abs(result[i] - size) > 1e-5)
				ok = false;

		printf("Multiplication test [%s]\n", ok ? "OK" : "FAILED");
		params.Free();
	}
}

void CudaTest()
{
	/****************************/
	//TESTS
	/****************************/
	CorrespondencesTest();
	MultiplicationTest();

	/****************************/
	//ALGORITHM
	/****************************/
	const auto testCloud = LoadCloud("data/rose.obj");
	const auto testCorrupted = LoadCloud("data/rose.obj");

	const auto hostBefore = CommonToThrustVector(testCloud);
	const auto hostAfter = CommonToThrustVector(testCorrupted);

	Cloud deviceCloudBefore = hostBefore;
	Cloud deviceCloudAfter = hostAfter;

	Cloud calculatedCloud(hostAfter.size());

	const auto scaleInput = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudBefore.begin(), scaleInput);
	const auto scaleInputCorrupted = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudAfter.begin(), deviceCloudAfter.end(), deviceCloudAfter.begin(), scaleInputCorrupted);

	const auto sampleTransform = glm::rotate(glm::translate(glm::mat4(1), { 0.05f, 0.05f, 0.05f }), glm::radians(5.f), { 0.5f, 0.5f, 0.5f });
	TransformCloud(deviceCloudAfter, deviceCloudAfter, sampleTransform);

	auto start = std::chrono::high_resolution_clock::now();
	const auto result = CudaICP(deviceCloudBefore, deviceCloudAfter);
	auto stop = std::chrono::high_resolution_clock::now();
	printf("Size: %d points, duration: %dms\n", testCloud.size(), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start));

	TransformCloud(deviceCloudBefore, calculatedCloud, result);

	Common::Renderer renderer(
		Common::ShaderType::SimpleModel,
		ThrustToCommonVector(deviceCloudBefore), //grey
		ThrustToCommonVector(deviceCloudAfter), //blue
		ThrustToCommonVector(calculatedCloud), //red
		std::vector<Point_f>(1));

	renderer.Show();
}
