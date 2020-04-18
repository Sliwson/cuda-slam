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
			auto p1 = thrust::get<0>(pair);
			auto p2 = thrust::get<1>(pair);
			return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
		}
	};
;
}

