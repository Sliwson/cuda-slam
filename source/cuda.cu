#include "cuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"

namespace CUDACommon
{
	void PrintVector(thrust::host_vector<float> vector)
	{
		for (int i = 0; i < vector.size(); i++)
		{
			printf("%f\n", vector[i]);
		}
	}

	void PrintVector(thrust::host_vector<glm::vec3> vector)
	{
		for (int i = 0; i < vector.size(); i++)
		{
			printf("%f %f %f\n", vector[i].x, vector[i].y, vector[i].z);
		}
	}

	void PrintVector(thrust::device_vector<float> vector)
	{
		thrust::host_vector<float> vec = vector;
		PrintVector(vec);
	}

	void PrintVector(thrust::device_vector<glm::vec3> vector)
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

	std::vector<Point_f> ThrustToCommonVector(const Cloud& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud = vec;
		std::vector<Point_f> outVector(vec.size());
		for (int i = 0; i < hostCloud.size(); i++)
			outVector[i] = { hostCloud[i].x, hostCloud[i].y, hostCloud[i].z };

		return outVector;
	}

	glm::vec3 CalculateCentroid(const Cloud& vec)
	{
		const auto sum = thrust::reduce(thrust::device, vec.begin(), vec.end());
		return sum / static_cast<float>(vec.size());
	}

	void TransformCloud(const Cloud& vec, Cloud& out, const glm::mat4& transform)
	{
		const auto functor = Functors::MatrixTransform(transform);
		thrust::transform(thrust::device, vec.begin(), vec.end(), out.begin(), functor);
	}

	__device__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
	{
		const auto d = second - first;
		return d.x * d.x + d.y * d.y + d.z * d.z;
	}

	float GetMeanSquaredError(const IndexIterator& permutation, const Cloud& before, const Cloud& after)
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

	void GetAlignedCloud(const Cloud& source, Cloud& target)
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

	glm::mat4 LeastSquaresSVD(const IndexIterator& permutation, const Cloud& before, const Cloud& after, Cloud& alignBefore, Cloud& alignAfter, CudaSvdParams params)
	{
		const int size = before.size();

		//align arrays
		const auto centroidBefore = CalculateCentroid(before);
		GetAlignedCloud(before, alignBefore);

		auto permutationIteratorBegin = thrust::make_permutation_iterator(after.begin(), permutation.begin());
		auto permutationIteratorEnd = thrust::make_permutation_iterator(after.end(), permutation.end());
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
		auto convertBefore = Functors::GlmToCuBlas(false, before.size(), params.workBefore);
		thrust::for_each(thrust::device, beforeZipBegin, beforeZipEnd, convertBefore);

		//multiply
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
		cusolverDnSgesvd(params.solverHandle, 'A', 'A', 3, 3, params.multiplyResult, 3, params.S, params.U, 3, params.VT, 3, params.work, params.workSize, nullptr, params.devInfo);
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
}
