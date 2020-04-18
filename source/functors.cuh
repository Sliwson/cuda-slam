#pragma once
#include "cuda.cuh"

namespace Functors {
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
		__device__ __host__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
		{
			const auto d = second - first;
			return d.x * d.x + d.y * d.y + d.z * d.z;
		}

		const glm::vec3* elementsAfter;
		int elementsSize;
	};
}

