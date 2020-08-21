#pragma once

#include "_common.h"

using namespace Common;

namespace Tests
{
	constexpr int RANDOM_SEED = 666;

	// Generate random data
	//
	float GetRandomFloat(float min, float max);
	Point_f GetRandomPoint(const Point_f& min, const Point_f& max);
	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count);
	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRange);
	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationAngle);
	glm::mat3 GetRandomRotationMatrix(float rotationRange);
	glm::vec3 GetRandomTranslationVector(float translationRange);

	// Helpers
	//
	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngle);
}