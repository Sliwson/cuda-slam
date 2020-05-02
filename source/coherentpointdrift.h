#include <utility>
#include <tuple>
#include "common.h"

namespace CoherentPointDrift
{
	std::pair<glm::mat3, glm::vec3> GetRigidCPDTransformationMatrix(
		const std::vector<Common::Point_f>& cloudBefore,
		const std::vector<Common::Point_f>& cloudAfter,
		int* iterations, 
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance);
}