#pragma once
#include <utility>
#include <tuple>
#include "common.h"

namespace Common {
	class Configuration;
}

namespace CoherentPointDrift
{
	std::pair<glm::mat3, glm::vec3> CalculateCpdWithConfiguration(
		const std::vector<Common::Point_f>& cloudBefore,
		const std::vector<Common::Point_f>& cloudAfter,
		Common::Configuration configuration);

	std::pair<glm::mat3, glm::vec3> GetRigidCPDTransformationMatrix(
		const std::vector<Common::Point_f>& cloudBefore,
		const std::vector<Common::Point_f>& cloudAfter,
		int* iterations, 
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance,
		Common::ApproximationType fgt);
}
