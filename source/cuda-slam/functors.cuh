#pragma once
#include "cudacommon.cuh"

namespace Functors
{
	struct MatrixTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		MatrixTransform(const glm::mat4& transform);

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector);

	private:
		glm::mat4 transformMatrix = glm::mat4(1.f);
	};

	struct ScaleTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		ScaleTransform(float multiplier);

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector);

	private:
		float multiplier = 1.f;
	};

	struct TranslateTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		TranslateTransform(glm::vec3 translation);

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector);

	private:
		glm::vec3 translation = { 0.f, 0.f, 0.f };
	};

	struct MSETransform : thrust::unary_function<thrust::tuple<glm::vec3, glm::vec3>, float>
	{
		__device__ __host__ float operator()(const thrust::tuple<glm::vec3, glm::vec3>& pair);
	};

	struct FindNearestIndex : thrust::unary_function<glm::vec3, int>
	{
		FindNearestIndex(const thrust::device_vector<glm::vec3>& elementsAfter);

		__device__ __host__ int operator()(const glm::vec3& vector);

	private:
		const glm::vec3* elementsAfter = nullptr;
		int elementsSize = 0;
	};

	struct GlmToCuBlas : thrust::unary_function<thrust::tuple<int, glm::vec3>, void>
	{
		GlmToCuBlas(bool transpose, int length, float* output);

		__device__ __host__ void operator()(const thrust::tuple<int, glm::vec3>& pair);

	private:
		bool transpose = false;
		float* output = nullptr;
		int length = 0;
	};

	struct CalculateSigmaSquaredInRow : thrust::unary_function<glm::vec3, float>
	{
		CalculateSigmaSquaredInRow(const thrust::device_vector<glm::vec3>& cloud);

		__device__ __host__ float operator()(const glm::vec3& vector);

	private:
		const glm::vec3* cloud = nullptr;
		int cloudSize = 0;
	};

	struct CalculateDenominator : thrust::unary_function<thrust::tuple<glm::vec3, int>, float>
	{
		CalculateDenominator(
			const glm::vec3& cloudBeforeItem,
			thrust::device_vector<float>& p,
			const float& multiplier,
			const bool& doTruncate,
			const float& truncate);

		__device__ __host__ float operator()(const thrust::tuple<glm::vec3, int>& vector);

	private:
		glm::vec3 cloudBeforeItem = glm::vec3(0.f);
		float* p = nullptr;
		float multiplier = 0.f;
		bool doTruncate = false;
		float truncate = 0.f;
	};

	struct CalculateP1AndPX : thrust::unary_function<int, void>
	{
		CalculateP1AndPX(
			const glm::vec3& cloudBeforeItem,
			const thrust::device_vector<float>& p,
			thrust::device_vector<float>& p1,
			thrust::device_vector<glm::vec3>& px,
			const float& denominator);

		__device__ __host__ void operator()(const int& index);

	private:
		glm::vec3 cloudBeforeItem = glm::vec3(0.f);
		const float* p = nullptr;
		float* p1 = nullptr;
		glm::vec3* px = nullptr;
		float denominator = 0.f;
	};

	struct CalculateSigmaSubtrahend : thrust::unary_function<thrust::tuple<int, float>, float>
	{
		CalculateSigmaSubtrahend(const float* pt1);

		__device__ __host__ float operator()(const thrust::tuple<int, float>& pair);

	private:
		const float* pt1 = nullptr;
	};
}
