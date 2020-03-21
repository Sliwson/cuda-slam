#include <stdio.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>

#include "common.h"
#include "loader.h"

namespace Common 
{
	namespace
	{
		constexpr float TEST_EPS = 1e-5f;

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

		Point_f TransformPoint(const Point_f& in, const glm::mat4& matrix)
		{
			const glm::vec3 result = matrix * glm::vec4(glm::vec3(in), 1.0f);
			return Point_f(result);
		}
	}

	std::vector<Point_f> LoadCloud(const std::string& path)
	{
		AssimpCloudLoader loader(path);
		if (loader.GetCloudCount() > 0)
			return loader.GetCloud(0);
		else
			return std::vector<Point_f>();
	}

	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count)
	{
		std::vector<Point_f> result;
		for (int i = 0; i < count; i++)
			result.push_back(GetRandomPoint(corner, corner + size));

		return result;
	}

	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRange)
	{
		const auto rotation = glm::rotate(glm::mat4(1.0f),
			GetRandomFloat(0, rotationRange),
			glm::vec3(GetRandomPoint(Point_f::Zero(), Point_f::One())));

		return glm::translate(rotation, glm::vec3(GetRandomPoint(translationMin, translationMax)));
	}

	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](Point_f p) { return TransformPoint(p, matrix); });
		return clone;
	}

	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation)
	{
		std::vector<Point_f> permutedCloud(input.size());
		for (int i = 0; i < input.size(); i++)
			permutedCloud[i] = input[permutation[i]];

		return permutedCloud;
	}

	Point_f GetCenterOfMass(const std::vector<Point_f> & cloud)
	{
		return std::accumulate(cloud.begin(), cloud.end(), Point_f::Zero()) / (float)cloud.size();
	}

	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		float diffSum = 0.0f;
		// We assume clouds are the same size but if error is significant, you might want to check it
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto transformed = TransformPoint(cloudBefore[i], matrix);
			const auto diff = cloudAfter[i] - transformed;
			diffSum += (diff.Length() * diff.Length());
		}

		return diffSum / cloudBefore.size();
	}

	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, matrix) <= TEST_EPS;
	}

	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, matrix);
	}

	bool TestPermutation(const std::vector<int>& expected, const std::vector<int>& actual)
	{
		return actual == expected;
	}

	std::vector<int> GetRandomPermutationVector(int size)
	{
		std::vector<int> permutation(size);
		std::iota(permutation.begin(), permutation.end(), 0);
		std::shuffle(permutation.begin(), permutation.end(), std::mt19937{ std::random_device{}() });
		return permutation;
	}

	void LibraryTest()
	{
		srand(666);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };

		//const auto cloud = GetRandomPointCloud(corner, size, cloudSize);
		const auto cloud = LoadCloud("data/bunny.obj");
		int cloudSize = cloud.size();

		const auto transform = GetRandomTransformMatrix({ -1, -1, -1 }, { 1, 1, 1 }, glm::radians(45.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);
		
		const auto calculatedPermutation = /* calculations here */ permutation;

		const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform);
		const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform);
		const auto resultPermutation = TestPermutation(permutation, calculatedPermutation);


		if (resultOrdered && resultUnordered && resultPermutation)
			printf("Library test [OK]\n");
		else
			printf("Library test [FAIL]\n");
	}
}
