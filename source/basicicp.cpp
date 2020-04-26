#include <Eigen/Dense>
#include <chrono>
#include <limits>

#include "basicicp.h"

using namespace Common;

namespace BasicICP
{
	std::pair<glm::mat3, glm::vec3> GetBasicICPTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>> correspondingPoints;
		std::vector<Point_f> transformedCloud = cloudBefore;

		while (maxIterations == -1 || *iterations < maxIterations)
		{
			// get corresponding points
			correspondingPoints = GetCorrespondingPoints(transformedCloud, cloudAfter, maxDistanceSquared);
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

			//printf("loop_nr %d, error: %f, correspondencesSize: %d\n", *iterations, *error, std::get<2>(correspondingPoints).size());

			if (*error < eps)
			{
				break;
			}

			(*iterations)++;
		}

		return std::make_pair(rotationMatrix, translationVector);
	}
}