#pragma once
#include "common.h"

using namespace Common;

namespace FastGaussTransform
{
	enum class ApproximationType;
}

namespace NonIterative
{
	enum class ApproximationType;
}

namespace Tests
{
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix, const float& testEps);
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix, const float& testEps);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps);

	void BasicICPTest(const char* objectPath, const int& pointCount, const float& testEps);
	void BasicICPTest(const char* objectPath1, const char* objectPath2, const int& pointCount1, const int& pointCount2, const float& testEps);

	void RigidCPDTest(
		const char* objectPath, 
		const int& pointCount,
		const float& testEps, 
		const float weight,
		const bool const_scale,
		const int max_iterations,
		const FastGaussTransform::ApproximationType fgt);
	void RigidCPDTest(
		const std::vector<Common::Point_f>& cloud,
		const float& testEps,
		const float weight,
		const bool const_scale,
		const int max_iterations,
		const FastGaussTransform::ApproximationType fgt);
	void RigidCPDTest(
		const char* objectPath1,
		const char* objectPath2,
		const int& pointCount1,
		const int& pointCount2,
		const float& testEps,
		const float weight,
		const bool const_scale,
		const int max_iterations,
		const FastGaussTransform::ApproximationType fgt);


	void NonIterativeTest(const char* objectPath, const int& pointCount, const float& testEps, const int& maxRepetitions, const NonIterative::ApproximationType& calculationType, const int& subcloudSize);
}