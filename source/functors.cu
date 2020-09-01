#include "functors.cuh"

namespace Functors
{
	namespace
	{
		__device__ __host__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
		{
			const auto d = second - first;
			return d.x * d.x + d.y * d.y + d.z * d.z;
		}
	}

	MatrixTransform::MatrixTransform(const glm::mat4& transform) : transformMatrix(transform) {}

	__device__ __host__ glm::vec3 MatrixTransform::operator()(const glm::vec3& vector)
	{
		return glm::vec3(transformMatrix * glm::vec4(vector, 1.f));
	}

	ScaleTransform::ScaleTransform(float multiplier) : multiplier(multiplier) {}

	__device__ __host__ glm::vec3 ScaleTransform::operator()(const glm::vec3& vector)
	{
		return multiplier * vector;
	}

	TranslateTransform::TranslateTransform(glm::vec3 translation) : translation(translation) {}

	__device__ __host__ glm::vec3 TranslateTransform::operator()(const glm::vec3& vector)
	{
		return vector + translation;
	}

	__device__ __host__ float MSETransform::operator()(const thrust::tuple<glm::vec3, glm::vec3>& pair)
	{
		const auto p1 = thrust::get<0>(pair);
		const auto p2 = thrust::get<1>(pair);
		const auto p = p2 - p1;
		return p.x * p.x + p.y * p.y + p.z * p.z;
	}

	FindNearestIndex::FindNearestIndex(const thrust::device_vector<glm::vec3>& elementsAfter)
	{
		this->elementsAfter = thrust::raw_pointer_cast(elementsAfter.data());
		this->elementsSize = elementsAfter.size();
	}

	__device__ __host__ int FindNearestIndex::operator()(const glm::vec3& vector)
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

	GlmToCuBlas::GlmToCuBlas(bool transpose, int length, float* output) : transpose(transpose), output(output), length(length) {}

	__device__ __host__ void GlmToCuBlas::operator()(const thrust::tuple<int, glm::vec3>& pair)
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

	CalculateSigmaSquaredInRow::CalculateSigmaSquaredInRow(const thrust::device_vector<glm::vec3>& cloud)
	{
		this->cloud = thrust::raw_pointer_cast(cloud.data());
		this->cloudSize = cloud.size();
	}

	__device__ __host__ float CalculateSigmaSquaredInRow::operator()(const glm::vec3& vector)
	{
		float sum = 0.0f;
		for (int i = 0; i < cloudSize; i++)
		{
			sum += GetDistanceSquared(vector, cloud[i]);
		}
		return sum;
	}

	CalculateDenominator::CalculateDenominator(
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

	__device__ __host__ float CalculateDenominator::operator()(const thrust::tuple<glm::vec3, int>& vector)
	{
		const float index = multiplier * GetDistanceSquared(cloudBeforeItem, vector.get<0>());
		//const float index = 3.0f;

		if (doTruncate && index < truncate)
		{
			p[vector.get<1>()] = 0.0f;
		}
		else
		{
			const float value = std::exp(index);
			p[vector.get<1>()] = value;
			return value;
		}
		return 0.0f;
	}

	CalculateP1AndPX::CalculateP1AndPX(
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

	__device__ __host__ void CalculateP1AndPX::operator()(const int& index)
	{
		if (p[index] != 0.0f)
		{
			const float value = p[index] / denominator;
			p1[index] += value;
			px[index] += cloudBeforeItem * value;
		}
	}

	CalculateSigmaSubtrahend::CalculateSigmaSubtrahend(const float* pt1) : pt1(pt1) { }

	__device__ __host__ float CalculateSigmaSubtrahend::operator()(const thrust::tuple<int, float>& pair)
	{
		const int idx = pair.get<0>();
		const float val = pair.get<1>();
		return val * val * pt1[idx / 3];
	}
}
