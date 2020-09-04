#pragma once
#include <utility>
#include <tuple>
#include "common.h"

namespace Common {
	struct Configuration;
}

namespace BasicICP
{
	std::pair<glm::mat3, glm::vec3> CalculateICPWithConfiguration(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, Common::Configuration config, int* iterations, float* error);
	std::pair<glm::mat3, glm::vec3> GetBasicICPTransformationMatrix(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations = -1, bool parallel = true);
}
