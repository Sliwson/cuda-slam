#pragma once

#include "cudacommon.cuh"

std::pair<glm::mat3, glm::vec3> GetCudaCpdTransformationMatrix(
	const std::vector<Common::Point_f>& cloudBefore,
	const std::vector<Common::Point_f>& cloudAfter,
	float eps,
	float weight,
	bool const_scale,
	int maxIterations,
	float tolerance,
	Common::ApproximationType fgt,
	int* iterations,
	float* error,
	const float& ratioOfFarField,
	const float& orderOfTruncation);
