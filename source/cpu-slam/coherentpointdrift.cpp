#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "coherentpointdrift.h"
#include "fgt.h"
#include "fgt_model.h"
#include "configuration.h"
#include "cpdutils.h"

using namespace Common;
using namespace FastGaussTransform;

namespace CoherentPointDrift
{
	float CalculateSigmaSquared(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);
	Probabilities ComputePMatrixFast(
		const std::vector<Point_f>& cloudTransformed,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const ApproximationType& fgt,
		const float& ratioOfFarField,
		const float& orderOfTruncation);
	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudTransformed,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate = false,
		float truncate = -1.0f);
	void MStep(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const Probabilities& probabilities,
		bool const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

	std::pair<glm::mat3, glm::vec3> CalculateCpdWithConfiguration(
		const std::vector<Common::Point_f>& cloudBefore,
		const std::vector<Common::Point_f>& cloudAfter,
		Common::Configuration config,
		int* iterations,
		float* error)
	{
		auto maxIterations = config.MaxIterations.has_value() ? config.MaxIterations.value() : -1;

		return GetRigidCPDTransformationMatrix(
			cloudBefore,
			cloudAfter,
			iterations,
			error, 
			config.ConvergenceEpsilon,
			config.CpdWeight,
			config.CpdConstScale, 
			maxIterations, 
			config.CpdTolerance,
			config.ApproximationType,
			config.RatioOfFarField,
			config.OrderOfTruncation);
	}

	//[0, 1, 2] if > 0, then use FGT. case 1: FGT with fixing sigma after it gets too small(faster, but the result can be rough)
	//case 2: FGT, followed by truncated Gaussian approximation(can be quite slow after switching to the truncated kernels, but more accurate than case 1)
	std::pair<glm::mat3, glm::vec3> GetRigidCPDTransformationMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		int* iterations,
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance,
		ApproximationType fgt,
		const float& ratioOfFarField,
		const float& orderOfTruncation)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		float scale = 1.0f;
		float sigmaSquared = CalculateSigmaSquared(cloudBefore, cloudAfter);
		float sigmaSquared_init = sigmaSquared;

		if (weight <= 0.0f)
			weight = 1e-6f;
		if (weight >= 1.0f)
			weight = 1.0f - 1e-6f;

		const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudBefore.size()) / ((1 - weight) * cloudAfter.size());
		float ntol = tolerance + 10.0f;
		float l = 0.0f;
		//TODO:
		//initialize memory for probabilities once
		Probabilities probabilities;
		std::vector<Point_f> transformedCloud = cloudBefore;
		//EM optimization
		while (*iterations < maxIterations && ntol > tolerance && sigmaSquared > eps)
		{
			//E-step
			if (fgt == ApproximationType::None)
				probabilities = ComputePMatrix(transformedCloud, cloudAfter, constant, sigmaSquared);
			else
				probabilities = ComputePMatrixFast(transformedCloud, cloudAfter, constant, weight, &sigmaSquared, sigmaSquared_init, fgt, ratioOfFarField, orderOfTruncation);

			ntol = std::abs((probabilities.error - l) / probabilities.error);
			l = probabilities.error;

			//M-step
			MStep(cloudBefore, cloudAfter, probabilities, const_scale, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			transformedCloud = GetTransformedCloud(cloudBefore, rotationMatrix, translationVector, scale);
			(*error) = sigmaSquared;
			(*iterations)++;
			printf("loop_nr %d, error: %f\n", *iterations, *error);
		}
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

	Probabilities ComputePMatrixFast(
		const std::vector<Point_f>& cloudTransformed,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const ApproximationType& fgt,
		const float& ratioOfFarField,
		const float& orderOfTruncation)
	{
		if (fgt == ApproximationType::Full)
		{
			if (*sigmaSquared < 0.05)
				*sigmaSquared = 0.05;
			return ComputePMatrixWithFGT(cloudTransformed, cloudAfter, weight, *sigmaSquared, sigmaSquaredInit, ratioOfFarField, orderOfTruncation);
		}
		if (fgt == ApproximationType::Hybrid)
		{
			if (*sigmaSquared > 0.015 * sigmaSquaredInit)
				return ComputePMatrixWithFGT(cloudTransformed, cloudAfter, weight, *sigmaSquared, sigmaSquaredInit, ratioOfFarField, orderOfTruncation);
			else
				return ComputePMatrix(cloudTransformed, cloudAfter, constant, *sigmaSquared, true, 1e-3f);
		}
		return Probabilities();
	}

	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudTransformed,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate)
	{
		const float multiplier = -0.5f / sigmaSquared;
		Eigen::VectorXf p = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf p1 = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf pt1 = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudTransformed.size(), DIMENSION);
		float error = 0.0;
		if (doTruncate)
			truncate = std::log(truncate);

		for (size_t x = 0; x < cloudAfter.size(); x++)
		{
			float denominator = 0.0;
			for (size_t k = 0; k < cloudTransformed.size(); k++)
			{
				const auto diffPoint = cloudAfter[x] - cloudTransformed[k];
				float index = multiplier * diffPoint.LengthSquared();

				if (doTruncate && index < truncate)
				{
					p(k) = 0.0f;
				}
				else
				{
					float value = std::exp(index);
					p(k) = value;
					denominator += value;
				}
			}
			denominator += constant;

			pt1(x) = 1.0f - constant / denominator;
			for (size_t k = 0; k < cloudTransformed.size(); k++)
			{
				if (p(k) != 0.0f)
				{
					float value = p(k) / denominator;
					p1(k) += value;
					px.row(k) += ConvertToEigenVector(cloudAfter[x]) * value;
				}
			}
			error -= std::log(denominator);
		}
		error += DIMENSION * cloudAfter.size() * std::log(sigmaSquared) / 2.0f;

		return { p1, pt1, px, error };
	}

	void MStep(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const Probabilities& probabilities,
		bool const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared)
	{
		const float Np = probabilities.p1.sum();
		const float InvertedNp = 1.0f / Np;
		auto EigenBefore = GetMatrix3XFromPointsVector(cloudBefore);
		auto EigenAfterT = GetMatrix3XFromPointsVector(cloudAfter);
		Eigen::Vector3f EigenCenterBefore = InvertedNp * EigenBefore * probabilities.p1;
		Eigen::Vector3f EigenCenterAfter = InvertedNp * EigenAfterT * probabilities.pt1;		

		const Eigen::MatrixXf AMatrix = (EigenBefore * probabilities.px).transpose() - Np * (EigenCenterAfter * EigenCenterBefore.transpose());

		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVT = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVT;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1.0f, 1.0f, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVT;

		const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		const float scaleNumerator = EigenScaleNumerator.trace();
		const float sigmaSubtrahend = (EigenAfterT.transpose().array().pow(2) * probabilities.pt1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterAfter.transpose() * EigenCenterAfter;
		const float scaleDenominator = (EigenBefore.transpose().array().pow(2) * probabilities.p1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterBefore.transpose() * EigenCenterBefore;

		if (const_scale == false)
		{
			*scale = scaleNumerator / scaleDenominator;
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / (float)DIMENSION;
		}
		else
		{
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend + scaleDenominator - 2 * scaleNumerator)) / (float)DIMENSION;
		}

		const Eigen::Vector3f EigenTranslationVector = EigenCenterAfter - (*scale) * EigenRotationMatrix * EigenCenterBefore;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);
	}
}
