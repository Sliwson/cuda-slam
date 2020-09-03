#pragma once

#include "cuda.cuh"

std::pair<glm::mat3, glm::vec3> GetCudaCpdTransformationMatrix(
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

