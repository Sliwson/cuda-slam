#pragma once

#include "cuda.cuh"

std::pair<glm::mat3, glm::vec3> GetCudaIcpTransformationMatrix(
	const std::vector<Common::Point_f>& cloudBefore,
	const std::vector<Common::Point_f>& cloudAfter,
	int* iterations,
	float eps,
	int maxIterations);
