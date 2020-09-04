#pragma once

#include "cuda.cuh"

std::pair<glm::mat3, glm::vec3> GetCudaNicpTransformationMatrix(
	const std::vector<Common::Point_f>& before,
	const std::vector<Common::Point_f>& after,
	float eps,
	int maxRepetitions,
	int batchSize,
	Common::ApproximationType approximationType,
	const int subcloudSize,
	int* repetitions,
	float* error
	);
