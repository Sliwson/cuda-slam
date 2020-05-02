#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "coherentpointdrift.h"

using namespace Common;

namespace CoherentPointDrift
{
	//typedef std::tuple<Eigen::VectorXf, Eigen::Vector3f, Eigen::MatrixXf> Probabilities;

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
		// Correspodences vector.
		std::vector<int> correspondence;
	};

	constexpr int DIMENSION = 3;
	float CalculateSigmaSquared(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);
	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared);
	void MStep(
		const Probabilities& probabilities,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		bool const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

	std::pair<glm::mat3, glm::vec3> GetRigidCPDTransformationMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		int* iterations,
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		float scale = 1.0f;
		float sigmaSquared = CalculateSigmaSquared(cloudBefore, cloudAfter);
		//TODO: add check for weight=1
		weight = std::clamp(weight, 0.0f, 1.0f);
		const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudAfter.size()) / ((1 - weight) * cloudBefore.size());
		float ntol = tolerance + 10.0f;
		float l = 0.0f;
		//std::vector<float> PMatrix = std::vector<float>(cloudBefore.size() * cloudAfter.size());
		Probabilities probabilities;
		std::vector<Point_f> transformedCloud = cloudAfter;
		//EM optimization
		while (*iterations < maxIterations && ntol>tolerance && sigmaSquared > eps)
		{
			//E-step
			probabilities = ComputePMatrix(cloudBefore, transformedCloud, constant, sigmaSquared);

			ntol = std::abs((probabilities.error - l) / probabilities.error);
			l = probabilities.error;

			//M-step
			MStep(probabilities, cloudBefore, cloudAfter, const_scale, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			transformedCloud = GetTransformedCloud(cloudAfter, rotationMatrix, translationVector, scale);

			printf("Iteration %d, sigmaSquared: %f, dL: %f, scale: %f\nTransformation Matrix:\n", *iterations, sigmaSquared, ntol, scale);
			PrintMatrix(ConvertToTransformationMatrix(scale * rotationMatrix, translationVector));
			printf("\n");

			(*iterations)++;
		}
		if(probabilities.correspondence.size() == transformedCloud.size())
			*error = GetMeanSquaredError(cloudBefore, transformedCloud, probabilities.correspondence);
		return std::make_pair(scale * rotationMatrix, translationVector);
	}

	float CalculateSigmaSquared(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < cloudBefore.size(); i++)
		{
			for (size_t j = 0; j < cloudAfter.size(); j++)
			{
				const auto diff = cloudBefore[i] - cloudAfter[j];
				sum += diff.LengthSquared();
			}
		}
		sum /= (float)(DIMENSION * cloudBefore.size() * cloudAfter.size());
		return sum;
	}

	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared)
	{
		const float multiplier = -0.5f / sigmaSquared;
		Eigen::VectorXf p = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::VectorXf p1 = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::VectorXf p1_max = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::VectorXf pt1 = Eigen::VectorXf::Zero(cloudBefore.size());
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudAfter.size(), DIMENSION);
		std::vector<int> correspondece = std::vector<int>(cloudAfter.size());
		float error = 0.0;
		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			float denominator = 0.0;
			for (size_t k = 0; k < cloudAfter.size(); k++)
			{
				const auto diffPoint = cloudBefore[x] - cloudAfter[k];				
				float index = multiplier * diffPoint.LengthSquared();
				float value = std::exp(index);
				p(k) = value;
				denominator += value;
			}
			denominator += constant;
			pt1(x) = 1.0f - constant / denominator;
			for (size_t k = 0; k < cloudAfter.size(); k++)
			{
				float value = p(k) / denominator;
				p1(k) += value;
				px.row(k) += ConvertToEigenVector(cloudBefore[x]) * value;
				if (value > p1_max(k))
				{
					correspondece[k] = x;
					p1_max(k) = value;
				}
			}
			error -= std::log(denominator);
		}
		error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2;

		return { p1, pt1, px, error, correspondece };
	}

	void MStep(
		const Probabilities& probabilities,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		bool const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared)
	{
		const float Np = probabilities.p1.sum();
		const float InvertedNp = 1.0f / Np;
		auto EigenBeforeT = GetMatrix3XFromPointsVector(cloudBefore);
		auto EigenAfterT = GetMatrix3XFromPointsVector(cloudAfter);
		Eigen::Vector3f EigenCenterBefore = InvertedNp * EigenBeforeT * probabilities.pt1;
		Eigen::Vector3f EigenCenterAfter = InvertedNp * EigenAfterT * probabilities.p1;

		const Eigen::MatrixXf AMatrix = (EigenAfterT * probabilities.px).transpose() - Np * (EigenCenterBefore * EigenCenterAfter.transpose());

		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVT = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVT;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1.0f, 1.0f, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVT;

		const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		const float scaleNumerator = EigenScaleNumerator.trace();
		const float sigmaSubtrahend = (EigenBeforeT.transpose().array().pow(2) * probabilities.pt1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterBefore.transpose() * EigenCenterBefore;
		const float scaleDenominator = (EigenAfterT.transpose().array().pow(2) * probabilities.p1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterAfter.transpose() * EigenCenterAfter;

		if (const_scale == false)
		{
			*scale = scaleNumerator / scaleDenominator;
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / (float)DIMENSION;
		}
		else
		{
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend + scaleDenominator - 2 * scaleNumerator)) / (float)DIMENSION;
		}

		const Eigen::Vector3f EigenTranslationVector = EigenCenterBefore - (*scale) * EigenRotationMatrix * EigenCenterAfter;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);
	}
}