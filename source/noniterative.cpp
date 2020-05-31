#include <Eigen/Dense>

#include "noniterative.h"
#include "timer.h"

using namespace Common;

namespace NonIterative
{
	std::pair<glm::mat3, glm::vec3> GetSingleNonIterativeSlamResult(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float* error, const std::vector<Point_f>& subcloudForComparison, float maxDistanceForComparison)
	{
		Timer timer("timer");
		timer.StartStage("SVD");

		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		std::vector<Point_f> transformedCloud = cloudBefore;

		Point_f centerBefore = GetCenterOfMass(cloudBefore);
		Point_f centerAfter = GetCenterOfMass(cloudAfter);

		std::vector<Point_f> alignedBefore = GetAlignedCloud(cloudBefore, centerBefore);
		std::vector<Point_f> alignedAfter = GetAlignedCloud(cloudAfter, centerAfter);

		Eigen::Matrix3Xf matrixBefore = GetMatrix3XFromPointsVector(alignedBefore);
		Eigen::Matrix3Xf matrixAfter = GetMatrix3XFromPointsVector(alignedAfter);

		//const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdBefore = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixBefore, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdBefore = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixBefore, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixBeforeTransposed = svdBefore.matrixU().transpose();

		//const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdAfter = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixAfter, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdAfter = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixAfter, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixAfter = svdAfter.matrixU();

		Eigen::Matrix3f rotation = uMatrixAfter * uMatrixBeforeTransposed;

		rotationMatrix = ConvertRotationMatrix(rotation);
		translationVector = glm::vec3(centerAfter) - (rotationMatrix * centerBefore);
		timer.StopStage("SVD");

		timer.StartStage("Correspon");
		float oldError = GetMeanSquaredError(alignedBefore, alignedAfter, rotationMatrix);
		std::vector<Point_f> transformedSubcloud = GetTransformedCloud(subcloudForComparison, rotationMatrix, translationVector);
		CorrespondingPointsTuple correspondingPoints = GetCorrespondingPoints(transformedSubcloud, cloudAfter, maxDistanceForComparison);
		*error = GetMeanSquaredError(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));
		printf("Error: %f (old: %f)\n", *error, oldError);
		timer.StopStage("Correspon");

		timer.PrintResults();
		return std::make_pair(rotationMatrix, translationVector);
	}

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float* error, float eps, int maxRepetitions)
	{
		int cloudSize = std::min(cloudBefore.size(), cloudAfter.size());
		std::pair<glm::mat3, glm::vec3> transformationResult;

		std::pair<glm::mat3, glm::vec3> bestTransformation;
		float minError = std::numeric_limits<float>::max();

		std::vector<int> subcloudIndices = GetRandomPermutationVector(1000);
		std::vector<Point_f> subcloudVertices = GetSubcloud(cloudBefore, subcloudIndices);
		assert(subcloudVertices.size() == 1000);
		const float maxDistanceForComparison = 1e6;

		for (int i = 0; i < maxRepetitions; i++)
		{
			const auto permutation = GetRandomPermutationVector(cloudSize);
			const auto permutedBefore = ApplyPermutation(cloudBefore, permutation);
			const auto permutedAfter = ApplyPermutation(cloudAfter, permutation);

			transformationResult = GetSingleNonIterativeSlamResult(permutedBefore, permutedAfter, error, subcloudVertices, maxDistanceForComparison);
			if (*error < minError)
			{
				minError = *error;
				bestTransformation = transformationResult;

				if (minError <= eps)
				{
					printf("Iterations: %d\n", i);
					return bestTransformation;
				}
			}
		}

		*error = minError;
		printf("Iterations: %d\n", maxRepetitions);
		return bestTransformation;
	}
}