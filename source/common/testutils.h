#pragma once

#include "_common.h"
#include "testrunner.h"

namespace Common {
	struct Configuration;
}

namespace Tests
{
	constexpr int RANDOM_SEED = 666;
	using Point_f = Common::Point_f;
	using AcquireFunc = std::function<std::vector<Common::Configuration>(Common::ComputationMethod)>;

	// Generate random data
	//
	float GetRandomFloat(float min, float max);
	Point_f GetRandomPoint(const Point_f& min, const Point_f& max);
	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count);
	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRadians);
	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationRadians);
	glm::mat3 GetRandomRotationMatrix(float rotationRadians);
	glm::vec3 GetRandomTranslationVector(float translation);

	// Helpers
	//
	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngle);

	// Run test batches, empty metdhos vector means running all methods
	//
	void RunTestSet(const AcquireFunc& acquireFunc, const Common::SlamFunc& slamFunc, const std::string& name, const std::vector<Common::ComputationMethod>& methods = {});
}