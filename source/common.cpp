#include <stdio.h>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/glm.hpp>
#include "common.h"

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

	bool TestTransform(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;

		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto transformed = TransformPoint(cloudBefore[i], matrix);
			const auto diff = cloudAfter[i] - transformed;
			if (std::abs(diff.Length()) > TEST_EPS)
				return false;
		}

		return true;
	}
	
	void LibraryTest()
	{
		srand(666);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };

		const auto cloud = GetRandomPointCloud(corner, size, 1000);
		const auto transform = GetRandomTransformMatrix({ -1, -1, -1 }, { 1, 1, 1 }, glm::radians(45.f));
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto result = TestTransform(cloud, transformedCloud, transform);

		if (result)
			printf("Library test [OK]\n");
		else
			printf("Library test [FAIL]\n");
	}
}
