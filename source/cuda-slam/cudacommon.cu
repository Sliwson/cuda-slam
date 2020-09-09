#include "cudacommon.cuh"
#include "functors.cuh"
#include "svdparams.cuh"
#include "parallelsvdhelper.cuh"

using namespace Common;

namespace CUDACommon
{
	const char* _cudaGetErrorEnum(cusolverStatus_t error)
	{
		switch (error)
		{
		case CUSOLVER_STATUS_SUCCESS:
			return "CUSOLVER_SUCCESS";

		case CUSOLVER_STATUS_NOT_INITIALIZED:
			return "CUSOLVER_STATUS_NOT_INITIALIZED";

		case CUSOLVER_STATUS_ALLOC_FAILED:
			return "CUSOLVER_STATUS_ALLOC_FAILED";

		case CUSOLVER_STATUS_INVALID_VALUE:
			return "CUSOLVER_STATUS_INVALID_VALUE";

		case CUSOLVER_STATUS_ARCH_MISMATCH:
			return "CUSOLVER_STATUS_ARCH_MISMATCH";

		case CUSOLVER_STATUS_EXECUTION_FAILED:
			return "CUSOLVER_STATUS_EXECUTION_FAILED";

		case CUSOLVER_STATUS_INTERNAL_ERROR:
			return "CUSOLVER_STATUS_INTERNAL_ERROR";

		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
		}

		return "<unknown>";
	}

	inline void __cusolveSafeCall(cusolverStatus_t err, const char* file, const int line)
	{
		if (CUSOLVER_STATUS_SUCCESS != err) {
			fprintf(stderr, "CUSOLVE error in file '%s', line %d\n error %d: %s\n\n", __FILE__, __LINE__, err, _cudaGetErrorEnum(err));
		}
	}

	extern "C" void cusolveSafeCall(cusolverStatus_t err) { __cusolveSafeCall(err, __FILE__, __LINE__); }

	__device__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
	{
		const auto d = second - first;
		return d.x * d.x + d.y * d.y + d.z * d.z;
	}

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

	void PrintVector(const thrust::host_vector<float>& vector)
	{
		for (int i = 0; i < vector.size(); i++)
		{
			printf("%f\n", vector[i]);
		}
	}

	void PrintVector(const thrust::host_vector<glm::vec3>& vector)
	{
		for (int i = 0; i < vector.size(); i++)
		{
			printf("%f %f %f\n", vector[i].x, vector[i].y, vector[i].z);
		}
	}

	void PrintVector(const thrust::device_vector<float>& vector)
	{
		thrust::host_vector<float> vec = vector;
		PrintVector(vec);
	}

	void PrintVector(const thrust::device_vector<glm::vec3>& vector)
	{
		thrust::host_vector<glm::vec3> vec = vector;
		PrintVector(vec);
	}

	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud(vec.size());
		for (int i = 0; i < vec.size(); i++)
			hostCloud[i] = (glm::vec3)vec[i];

		return hostCloud;
	}

	std::vector<Point_f> ThrustToCommonVector(const GpuCloud& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud = vec;
		std::vector<Point_f> outVector(vec.size());
		for (int i = 0; i < hostCloud.size(); i++)
			outVector[i] = { hostCloud[i].x, hostCloud[i].y, hostCloud[i].z };

		return outVector;
	}

	glm::vec3 CalculateCentroid(const GpuCloud& vec)
	{
		const auto sum = thrust::reduce(thrust::device, vec.begin(), vec.end());
		return sum / static_cast<float>(vec.size());
	}

	void TransformCloud(const GpuCloud& vec, GpuCloud& out, const glm::mat4& transform)
	{
		const auto functor = Functors::MatrixTransform(transform);
		thrust::transform(thrust::device, vec.begin(), vec.end(), out.begin(), functor);
	}

	float GetMeanSquaredError(const IndexIterator& permutation, const GpuCloud& before, const GpuCloud& after)
	{
		auto permutationIteratorBegin = thrust::make_permutation_iterator(after.begin(), permutation.begin());
		auto permutationIteratorEnd = thrust::make_permutation_iterator(after.end(), permutation.end());
		auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(permutationIteratorBegin, before.begin()));
		auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(permutationIteratorEnd, before.end()));
		auto mseFunctor = Functors::MSETransform();
		auto sumFunctor = thrust::plus<float>();
		auto result = thrust::transform_reduce(thrust::device, zipBegin, zipEnd, mseFunctor, 0.f, sumFunctor);
		return result / after.size();
	}

	void GetAlignedCloud(const GpuCloud& source, GpuCloud& target)
	{
		const auto centroid = CalculateCentroid(source);
		const auto transform = Functors::TranslateTransform(-centroid);
		thrust::transform(thrust::device, source.begin(), source.end(), target.begin(), transform);
	}

	void CuBlasMultiply(float* A, float* B, float* C, int size, CudaSvdParams& params)
	{
		const float alpha = 1.f, beta = 0.f;
		cublasSgemm(params.multiplyHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, size, &alpha, A, 3, B, size, &beta, C, 3);
	}

	glm::mat3 CreateGlmMatrix(float* squareMatrix)
	{
		return glm::transpose(glm::make_mat3(squareMatrix));
	}

	glm::mat4 LeastSquaresSVD(const IndexIterator& permutation, const GpuCloud& before, const GpuCloud& after, GpuCloud& alignBefore, GpuCloud& alignAfter, CudaSvdParams params)
	{
		const int size = before.size();

		//align arrays
		const auto centroidBefore = CalculateCentroid(before);
		GetAlignedCloud(before, alignBefore);

		auto permutationIteratorBegin = thrust::make_permutation_iterator(after.begin(), permutation.begin());
		auto permutationIteratorEnd = thrust::make_permutation_iterator(after.end(), permutation.end());
		assert(permutationIteratorBegin + permutation.size() == permutationIteratorEnd);

		thrust::copy(thrust::device, permutationIteratorBegin, permutationIteratorEnd, alignAfter.begin());
		const auto centroidAfter = CalculateCentroid(alignAfter);
		GetAlignedCloud(alignAfter, alignAfter);

		//create array AFTER (transposed)
		auto countingBegin = thrust::make_counting_iterator<int>(0);
		auto countingEnd = thrust::make_counting_iterator<int>(alignAfter.size());
		auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(countingBegin, alignAfter.begin()));
		auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(countingEnd, alignAfter.end()));

		auto convertAfter = Functors::GlmToCuBlas(true, size, params.workAfter);
		thrust::for_each(thrust::device, zipBegin, zipEnd, convertAfter);

		//create array BEFORE
		const auto beforeZipBegin = thrust::make_zip_iterator(thrust::make_tuple(countingBegin, alignBefore.begin()));
		const auto beforeZipEnd = thrust::make_zip_iterator(thrust::make_tuple(countingEnd, alignBefore.end()));
		auto convertBefore = Functors::GlmToCuBlas(false, size, params.workBefore);
		thrust::for_each(thrust::device, beforeZipBegin, beforeZipEnd, convertBefore);

		//multiply
		assert(beforeZipEnd - beforeZipBegin == zipEnd - zipBegin);
		CuBlasMultiply(params.workBefore, params.workAfter, params.multiplyResult, size, params);
		float result[9];
		cudaMemcpy(result, params.multiplyResult, 9 * sizeof(float), cudaMemcpyDeviceToHost);
		auto matrix = CreateGlmMatrix(result);
		//return Common::GetTransform(matrix, centroidBefore, centroidAfter);

		float transposed[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				transposed[3 * i + j] = result[3 * j + i];
		cudaMemcpy(params.multiplyResult, transposed, 9 * sizeof(float), cudaMemcpyHostToDevice);

		//svd
		cusolveSafeCall(cusolverDnSgesvd(params.solverHandle, 'A', 'A', params.m, params.n, params.multiplyResult, params.m, params.S, params.U, params.m, params.VT, params.n, params.work, params.workSize, nullptr, params.devInfo));
		int svdResultInfo = 0;
		cudaMemcpy(&svdResultInfo, params.devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		if (svdResultInfo != 0)
			printf("Svd execution failed!\n");

		float hostS[9], hostVT[9], hostU[9];
		const int copySize = 9 * sizeof(float);
		cudaMemcpy(hostS, params.S, copySize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostVT, params.VT, copySize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostU, params.U, copySize, cudaMemcpyDeviceToHost);

		auto gVT = glm::transpose(CreateGlmMatrix(hostVT));
		auto gU = glm::transpose(CreateGlmMatrix(hostU));

		//revert signs to match svd cpu solution
		for (int i = 0; i < 3; i++)
		{
			gU[1][i] = -gU[1][i];
			gVT[i][1] = -gVT[i][1];
		}

		const float determinant = glm::determinant(gU * gVT);
		const auto diagonal = glm::diagonal3x3(glm::vec3{ 1.f, 1.f, determinant });
		const auto rotation = gU * diagonal * gVT;

		const auto translation = centroidAfter - rotation * centroidBefore;

		auto transformation = glm::mat4(0.f);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				transformation[i][j] = rotation[i][j];

		transformation[3][0] = translation.x;
		transformation[3][1] = translation.y;
		transformation[3][2] = translation.z;
		transformation[3][3] = 1.0f;

		return transformation;
	}

	void ApplyPermutation(const GpuCloud& inputCloud, IndexIterator permutation, GpuCloud& outputCloud)
	{
		assert(outputCloud.size() == inputCloud.size());

		int permutationSize = permutation.size();
		if (permutationSize < inputCloud.size())
		{
			permutation.resize(inputCloud.size());
			auto helperCountingIterator = thrust::make_counting_iterator(0);
			thrust::copy(helperCountingIterator + permutationSize, helperCountingIterator + inputCloud.size(), permutation.begin() + permutationSize);
		}

		auto permutationIterBegin = thrust::make_permutation_iterator(inputCloud.begin(), permutation.begin());
		auto permutationIterEnd = thrust::make_permutation_iterator(inputCloud.end(), permutation.end());
		thrust::copy(permutationIterBegin, permutationIterEnd, outputCloud.begin());
	}

	void GetCorrespondingPoints(thrust::device_vector<int>& indices, const GpuCloud& before, const GpuCloud& after)
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

		CudaSvdParams params(size, size, 3, 3);
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
