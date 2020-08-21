#include <Eigen/Dense>
#include <chrono>
#include <limits>

#include "basicicp.h"
#include "configuration.h"

using namespace Common;

namespace BasicICP
{
	constexpr float ICP_EPS = 1e-5;

	std::pair<glm::mat3, glm::vec3> CalculateICPWithConfiguration(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, Common::Configuration config)
	{
		auto maxIterations = config.MaxIterations.has_value() ? config.MaxIterations.value() : -1;

		auto parallel = config.ExecutionPolicy.has_value() ?
			config.ExecutionPolicy.value() == Common::ExecutionPolicy::Parallel :
			true;

		int iterations = 0;
		float error = 0;

		auto result = GetBasicICPTransformationMatrix(cloudBefore, cloudAfter, &iterations, &error, ICP_EPS, config.MaxDistanceSquared, maxIterations, parallel);
		return result;
	}

	std::pair<glm::mat3, glm::vec3> GetBasicICPTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations, bool parallel)
	{
		constexpr auto iterationTreshold = 50;

		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>> correspondingPoints;
		std::vector<Point_f> transformedCloud = cloudBefore;

		while ((maxIterations == -1 && *iterations < iterationTreshold) || *iterations < maxIterations)
		{
			// get corresponding points
			correspondingPoints = GetCorrespondingPoints(transformedCloud, cloudAfter, maxDistanceSquared, parallel);
			if (std::get<0>(correspondingPoints).size() == 0)
				break;

			// use svd
			auto transformationMatrix = LeastSquaresSVD(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));

			// update rotation matrix and translation vector
			rotationMatrix = transformationMatrix.first * rotationMatrix;
			translationVector = transformationMatrix.second + translationVector;

			transformedCloud = GetTransformedCloud(cloudBefore, rotationMatrix, translationVector);
			// count error
			*error = GetMeanSquaredError(transformedCloud, cloudAfter, std::get<2>(correspondingPoints), std::get<3>(correspondingPoints));

			printf("loop_nr %d, error: %f, correspondencesSize: %d\n", *iterations, *error, std::get<2>(correspondingPoints).size());

			if (*error < eps)
			{
				break;
			}

			(*iterations)++;
		}

		return std::make_pair(rotationMatrix, translationVector);
	}
}
