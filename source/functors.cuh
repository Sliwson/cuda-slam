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
}

