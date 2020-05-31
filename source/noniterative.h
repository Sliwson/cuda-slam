#pragma once
#include <utility>
#include <tuple>
#include "common.h"

using namespace Common;

namespace NonIterative
{
	std::pair<glm::mat3, glm::vec3> GetSingleNonIterativeSlamResult(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float* error, const std::vector<Point_f>& subcloudForComparison, float maxDistanceForComparison);
	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float* error, float eps, int maxRepetitions);
}