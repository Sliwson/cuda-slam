#include "testutils.h"

namespace Tests
{
	// Generate random data
	//
	float GetRandomFloat(float min, float max)
	{
		float range = max - min;
		return static_cast<float>(rand()) / RAND_MAX * range + min;
	}

	Point_f GetRandomPoint(const Point_f& min, const Point_f& max)
	{
		return {
			GetRandomFloat(min.x, max.x),
			GetRandomFloat(min.y, max.y),
			GetRandomFloat(min.z, max.z)
		};
	}

	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count)
	{
		std::vector<Point_f> result;
		for (int i = 0; i < count; i++)
			result.push_back(GetRandomPoint(corner, corner + size));
		return result;
	}

	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRangeRadians)
	{
		const auto rotation = glm::mat4(GetRandomRotationMatrix(rotationRangeRadians));
		return glm::translate(rotation, glm::vec3(GetRandomPoint(translationMin, translationMax)));
	}

	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationAngleRadians)
	{
		const auto rotation = glm::mat4(GetRotationMatrix(rotationAxis, rotationAngleRadians));
		return glm::translate(rotation, glm::vec3(translation));
	}

	glm::mat3 GetRandomRotationMatrix(float rotationRangeRadians)
	{
		const auto angle = GetRandomFloat(0, rotationRangeRadians);
		//const auto angle = glm::radians(30.0f);
		const auto rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(GetRandomPoint(Point_f::Zero(), Point_f::One())));
		return glm::mat3(rotation);
	}

	std::vector<int> GetRandomPermutationVector(int size)
	{
		std::vector<int> permutation(size);
		std::iota(permutation.begin(), permutation.end(), 0);
		std::shuffle(permutation.begin(), permutation.end(), std::mt19937{ std::random_device{}() });
		return permutation;
	}


	// Helpers
	//
	std::vector<int> InversePermutation(const std::vector<int>& permutation)
	{
		auto inversedPermutation = std::vector<int>(permutation.size());
		for (int i = 0; i < permutation.size(); i++)
		{
			inversedPermutation[permutation[i]] = i;
		}
		return inversedPermutation;
	}

	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngleRadians)
	{
		const auto rotation = glm::rotate(glm::mat4(1.0f), rotationAngleRadians, glm::normalize(glm::vec3(rotationAxis)));
		return glm::mat3(rotation);
	}

	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation)
	{
		std::vector<Point_f> permutedCloud(input.size());
		for (int i = 0; i < input.size(); i++)
			permutedCloud[i] = input[permutation[i]];
		return permutedCloud;
	}
}