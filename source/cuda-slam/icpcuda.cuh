#pragma once

#include "cuda.cuh"

std::pair<glm::mat3, glm::vec3> GetCudaIcpTransformationMatrix(
	const std::vector<Common::Point_f>& cloudBefore,
	const std::vector<Common::Point_f>& cloudAfter,
	float eps,
	int maxIterations,
	int* iterations,
	float* error);
