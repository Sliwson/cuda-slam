#pragma once
#include "cuda.cuh"

namespace Functors {
	namespace {
		__device__ __host__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
		{
			const auto d = second - first;
			return d.x * d.x + d.y * d.y + d.z * d.z;
		}
	}

	struct MatrixTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		MatrixTransform(const glm::mat4& transform) : transformMatrix(transform) {}

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector)
		{
			return glm::vec3(transformMatrix * glm::vec4(vector, 1.f));
		}

	private:
		glm::mat4 transformMatrix = glm::mat4(1.f);
	};

	struct ScaleTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		ScaleTransform(float multiplier) : multiplier(multiplier) {}

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector)
		{
			return multiplier * vector;
		}

	private:
		float multiplier = 1.f;
	};

	struct TranslateTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		TranslateTransform(glm::vec3 translation) : translation(translation) {}

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector)
		{
			return vector + translation;
		}

	private:
		glm::vec3 translation = { 0.f, 0.f, 0.f };
	};

	struct MSETransform : thrust::unary_function<thrust::tuple<glm::vec3, glm::vec3>, float>
	{
		__device__ __host__ float operator()(const thrust::tuple<glm::vec3, glm::vec3>& pair)
		{
			const auto p1 = thrust::get<0>(pair);
			const auto p2 = thrust::get<1>(pair);
			const auto p = p2 - p1;
			return p.x * p.x + p.y * p.y + p.z * p.z;
		}
	};

	struct FindNearestIndex : thrust::unary_function<glm::vec3, int>
	{
		FindNearestIndex(const thrust::device_vector<glm::vec3>& elementsAfter)
		{
			this->elementsAfter = thrust::raw_pointer_cast(elementsAfter.data());
			this->elementsSize = elementsAfter.size();
		}

		__device__ __host__ int operator()(const glm::vec3& vector)
		{
			if (elementsSize == 0)
				return 0;

			int nearestIdx = 0;
			float smallestError = GetDistanceSquared(vector, elementsAfter[0]);
			for (int i = 1; i < elementsSize; i++)
			{
				const auto dist = GetDistanceSquared(vector, elementsAfter[i]);
				if (dist < smallestError)
				{
					smallestError = dist;
					nearestIdx = i;
				}
			}

			return nearestIdx;
		}

	private:
		const glm::vec3* elementsAfter;
		int elementsSize;
	};

	struct GlmToCuBlas : thrust::unary_function<thrust::tuple<int, glm::vec3>, void>
	{
		GlmToCuBlas(bool transpose, int length, float* output) : transpose(transpose), output(output), length(length) { }

		__device__ __host__ void operator()(const thrust::tuple<int, glm::vec3>& pair)
		{
			const auto idx = thrust::get<0>(pair);
			const auto vector = thrust::get<1>(pair);

			if (transpose)
			{
				output[idx] = vector.x;
				output[idx + length] = vector.y;
				output[idx + 2 * length] = vector.z;
			}
			else
			{
				output[3 * idx] = vector.x;
				output[3 * idx + 1] = vector.y;
				output[3 * idx + 2] = vector.z;
			}
		}

	private:
		bool transpose = false;
		float* output = nullptr;
		int length = 0;
	};

	struct CalculateSigmaSquaredInRow : thrust::unary_function<glm::vec3, float>
	{
		CalculateSigmaSquaredInRow(const thrust::device_vector<glm::vec3>& cloud)
		{
			this->cloud = thrust::raw_pointer_cast(cloud.data());
			this->cloudSize = cloud.size();
		}

		__device__ __host__ float operator()(const glm::vec3& vector)
		{
			float sum = 0.0f;
			for (int i = 0; i < cloudSize; i++)
			{
				sum += GetDistanceSquared(vector, cloud[i]);
			}
			return sum;
		}

	private:
		__device__ __host__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
		{
			const auto d = second - first;
			return d.x * d.x + d.y * d.y + d.z * d.z;
		}

		const glm::vec3* cloud;
		int cloudSize;
	};

	struct CalculateDenominator : thrust::unary_function<thrust::tuple<glm::vec3, int>, float>
	{
		CalculateDenominator(
			const glm::vec3& cloudBeforeItem,
			thrust::device_vector<float>& p,
			const float& multiplier,
			const bool& doTruncate,
			const float& truncate) :
			cloudBeforeItem(cloudBeforeItem),
			multiplier(multiplier),
			doTruncate(doTruncate),
			truncate(truncate)
		{
			this->p = thrust::raw_pointer_cast(p.data());
		}

		__device__ __host__ float operator()(const thrust::tuple<glm::vec3, int>& vector)
		{
			float index = multiplier * GetDistanceSquared(cloudBeforeItem, vector.get<0>());

			if (doTruncate && index < truncate)
			{
				p[vector.get<1>()] = 0.0f;
				return 0.0f;
			}
			else
			{
				const float value = std::exp(index);
				p[vector.get<1>()] = value;
				return value;
			}
		}

	private:
		glm::vec3 cloudBeforeItem = glm::vec3(0.f);
		float* p;
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
			const float& denominator) :
			cloudBeforeItem(cloudBeforeItem),
			denominator(denominator)
		{
			this->p = thrust::raw_pointer_cast(p.data());
			this->p1 = thrust::raw_pointer_cast(p1.data());
			this->px = thrust::raw_pointer_cast(px.data());
		}

		__device__ __host__ void operator()(const int& index)
		{
			if (p[index] != 0.0f)
			{
				const float value = p[index] / denominator;
				p1[index] += value;
				px[index] += cloudBeforeItem * value;
			}
		}

	private:
		glm::vec3 cloudBeforeItem = glm::vec3(0.f);
		const float* p;
		float* p1;
		glm::vec3* px;
		float denominator = 0.f;
	};
}