#pragma once
#include "common.h"

using namespace Common;

namespace Tests
{
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix, const float& testEps);
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix, const float& testEps);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps);

	void BasicICPTest(const char* objectPath, const int& pointCount, const float& testEps);
	void NonIterativeTest(const char* objectPath, const int& pointCount, const float& testEps);
}