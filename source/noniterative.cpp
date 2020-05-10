#include <Eigen/Dense>

#include "noniterative.h"

using namespace Common;

namespace NonIterative
{
	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float* error)
	{
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>> correspondingPoints;
		std::vector<Point_f> transformedCloud = cloudBefore;

		Point_f centerBefore = GetCenterOfMass(cloudBefore);
		Point_f centerAfter = GetCenterOfMass(cloudAfter);

		std::vector<Point_f> alignedBefore = GetAlignedCloud(cloudBefore, centerBefore);
		std::vector<Point_f> alignedAfter = GetAlignedCloud(cloudAfter, centerAfter);

		Eigen::Matrix3Xf matrixBefore = GetMatrix3XFromPointsVector(alignedBefore);
		Eigen::Matrix3Xf matrixAfter = GetMatrix3XFromPointsVector(alignedAfter);

		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdBefore = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixBefore, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixBeforeTransposed = svdBefore.matrixU().transpose();

		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdAfter = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixAfter, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixAfter = svdAfter.matrixU();

		Eigen::Matrix3f rotation = uMatrixAfter * uMatrixBeforeTransposed;

		rotationMatrix = ConvertRotationMatrix(rotation);
		translationVector = glm::vec3(centerAfter) - (rotationMatrix * centerBefore);

		//TODO: Implement error calculations
		//*error = GetMeanSquaredError()
		return std::make_pair(rotationMatrix, translationVector);
	}
}