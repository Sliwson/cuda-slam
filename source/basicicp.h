#pragma once
#include <utility>
#include <tuple>
#include "common.h"

namespace BasicICP
{
	std::pair<glm::mat3, glm::vec3> GetBasicICPTransformationMatrix(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations = -1);
}