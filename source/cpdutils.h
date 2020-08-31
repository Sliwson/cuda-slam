#pragma once

#include "coherentpointdrift.h"

namespace CoherentPointDrift
{
	struct Probabilities
	{
		// The probability matrix, multiplied by the identity vector.
		Eigen::VectorXf p1;
		// The probability matrix, transposed, multiplied by the identity vector.
		Eigen::VectorXf pt1;
		// The probability matrix multiplied by the fixed(cloud before) points.
		Eigen::MatrixXf px;
		// The total error.
		float error;
	};
}

namespace CPDutils
{
	CoherentPointDrift::Probabilities ComputePMatrixWithFGT(
		const std::vector<Common::Point_f>& cloudTransformed,
		const std::vector<Common::Point_f>& cloudAfter,
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit);
}
