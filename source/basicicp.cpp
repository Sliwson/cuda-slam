#include <Eigen/Dense>
#include <chrono>
#include <limits>

#include "basicicp.h"

using namespace Common;

namespace BasicICP
{
	std::pair<glm::mat3, glm::vec3> BasicICP(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float eps, float maxDistanceSquared, int maxIterations)
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

	std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>>GetCorrespondingPoints(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared)
	{
		std::vector<Point_f> correspondingFromCloudBefore(cloudBefore.size());
		std::vector<Point_f> correspondingFromCloudAfter(cloudBefore.size());
		std::vector<int> correspondingIndexesBefore(cloudBefore.size());
		std::vector<int> correspondingIndexesAfter(cloudBefore.size());
		int correspondingCount = 0;

		for (int i = 0; i < cloudBefore.size(); i++)
		{
			int closestIndex = -1;
			float closestDistance = std::numeric_limits<float>::max();

			for (int j = 0; j < cloudAfter.size(); j++)
			{
				float distance = (cloudAfter[j] - cloudBefore[i]).LengthSquared();

				if (distance < closestDistance || closestIndex == -1)
				{
					closestDistance = distance;
					closestIndex = j;
				}
			}
			if (closestDistance < maxDistanceSquared && closestIndex >= 0)
			{
				correspondingFromCloudBefore[correspondingCount] = cloudBefore[i];
				correspondingFromCloudAfter[correspondingCount] = cloudAfter[closestIndex];
				correspondingIndexesBefore[correspondingCount] = i;
				correspondingIndexesAfter[correspondingCount] = closestIndex;

				correspondingCount++;
			}
		}

		correspondingFromCloudBefore.resize(correspondingCount);
		correspondingFromCloudAfter.resize(correspondingCount);
		correspondingIndexesBefore.resize(correspondingCount);
		correspondingIndexesAfter.resize(correspondingCount);

		return std::make_tuple(correspondingFromCloudBefore, correspondingFromCloudAfter, correspondingIndexesBefore, correspondingIndexesAfter);
	}

	std::pair<glm::mat3, glm::vec3> LeastSquaresSVD(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		Eigen::Matrix3f rotationMatrix;// = Eigen::Matrix3f::Identity();
		Eigen::Vector3f translationVector;// = Eigen::Vector3f::Zero();

		auto centerBefore = GetCenterOfMass(cloudBefore);
		auto centerAfter = GetCenterOfMass(cloudAfter);

		const Eigen::Matrix3Xf alignedBefore = GetMatrix3XFromPointsVector(GetAlignedCloud(cloudBefore, centerBefore));
		const Eigen::Matrix3Xf alignedAfter = GetMatrix3XFromPointsVector(GetAlignedCloud(cloudAfter, centerAfter));

		//version with thinU (docs say it should be enough for us, further efficiency tests are needed
		// cheching option with MatrixXf instead od Matrix3f because documantation says: my matrix should have dynamic number of columns
		const Eigen::MatrixXf matrix = alignedAfter * alignedBefore.transpose();
		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		//version with fullU
		//const Eigen::Matrix3f matrix = alignedAfter * alignedBefore.transpose();
		//const Eigen::JacobiSVD<Eigen::Matrix3f> svd = Eigen::JacobiSVD<Eigen::Matrix3f>(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVtransposed = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVtransposed;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant());

		rotationMatrix = matrixU * diag * matrixVtransposed;

		glm::mat3 rotationMatrixGLM = ConvertRotationMatrix(rotationMatrix);

		glm::vec3 translationVectorGLM = glm::vec3(centerAfter) - (rotationMatrixGLM * centerBefore);

		return std::make_pair(rotationMatrixGLM, translationVectorGLM);
	}
}