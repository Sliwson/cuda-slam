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

	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRadians)
	{
		const auto rotation = glm::mat4(GetRandomRotationMatrix(rotationRadians));
		return glm::translate(rotation, glm::vec3(GetRandomPoint(translationMin, translationMax)));
	}

	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationRadians)
	{
		const auto rotation = glm::mat4(GetRotationMatrix(rotationAxis, rotationRadians));
		return glm::translate(rotation, glm::vec3(translation));
	}

	glm::mat3 GetRandomRotationMatrix(float rotationRadians)
	{
		const auto axis = glm::vec3(GetRandomPoint(Point_f::Zero(), Point_f::One()));
		const auto rotation = glm::rotate(glm::mat4(1.0f), rotationRadians, glm::normalize(axis));
		return glm::mat3(rotation);
	}

	glm::vec3 GetRandomTranslationVector(float translation)
	{
		const auto point = GetRandomPoint({ -1.f, -1.f, -1.f }, { 1.f, 1.f, 1.f });
		const auto normalized = point / point.Length();
		return normalized * translation;
	}


	// Helpers
	//
	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngleRadians)
	{
		const auto rotation = glm::rotate(glm::mat4(1.0f), rotationAngleRadians, glm::normalize(glm::vec3(rotationAxis)));
		return glm::mat3(rotation);
	}
}
