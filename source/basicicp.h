#pragma once
#include <utility>
#include <tuple>
#include "common.h"

namespace BasicICP
{
	std::pair<glm::mat3, glm::vec3> BasicICP(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations = -1);
	std::tuple<std::vector<Common::Point_f>, std::vector<Common::Point_f>, std::vector<int>, std::vector<int>>GetCorrespondingPoints(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, float maxDistanceSquared);
	std::pair<glm::mat3, glm::vec3> LeastSquaresSVD(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter);
}