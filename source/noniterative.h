#pragma once
#include <utility>
#include <tuple>
#include "common.h"
#include "configuration.h"

using namespace Common;

namespace Common
{
	class NonIterativeSlamResult;
}

namespace NonIterative
{
	std::pair<glm::mat3, glm::vec3> CalculateNonIterativeWithConfiguration(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		Common::Configuration config);

	NonIterativeSlamResult GetSingleNonIterativeSlamResult(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter);

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(
		const std::vector<Point_f>& cloudBefore, 
		const std::vector<Point_f>& cloudAfter, 
		int* repetitions,
		float* error, 
		float eps, 
		int maxRepetitions, 
		const ApproximationType& calculationType, 
		bool parallel = false,
		int subcloudSize = -1);
}