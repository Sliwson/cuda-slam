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
	};

	constexpr int DIMENSION = 3;
	float CalculateSigmaSquared(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);
	void ComputePMatrix(
		std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale);
	void ComputePMatrix2(
		std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared);
	Probabilities ComputePMatrix3(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared);
	void MStep(
		const std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);
	void MStep2(
		const std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);
	void MStep3(
		const Probabilities& probabilities,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

	std::pair<glm::mat3, glm::vec3> GetRigidCPDTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float eps, float weight, int maxIterations)
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
		
		//std::vector<float> PMatrix = std::vector<float>(cloudBefore.size() * cloudAfter.size());
		Probabilities probabilities;
		std::vector<Point_f> transformedCloud = cloudAfter;
		//EM optimization
		while (maxIterations == -1 || *iterations < maxIterations)
		{
			//E-step
			//ComputePMatrix2(PMatrix, cloudBefore, transformedCloud, constant, sigmaSquared);
			probabilities = ComputePMatrix3(cloudBefore, transformedCloud, constant, sigmaSquared);

			//M-step
			MStep3(probabilities, cloudBefore, cloudAfter, &rotationMatrix, &translationVector, &scale, &sigmaSquared);
			//MStep(PMatrix, cloudBefore, cloudAfter, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			//if doesnt work change this line
			transformedCloud = GetTransformedCloud(cloudAfter, rotationMatrix, translationVector, scale);


			printf("Iteration %d, sigmaSquared: %f, scale: %f\nTransformation Matrix:\n", *iterations, sigmaSquared, scale);
			PrintMatrix(ConvertToTransformationMatrix(scale * rotationMatrix, translationVector));
			printf("\n");

			if (sigmaSquared < eps)
				break;

			(*iterations)++;
			//think about convergence (something with PMatrix)
		}
		//rotationMatrix *= scale;
		//
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

	void ComputePMatrix(
		std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant, 
		const float& sigmaSquared,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale)
	{
		const float multiplier = -0.5f / sigmaSquared;
		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			float denominator = constant;			
			for (size_t k = 0; k < cloudAfter.size(); k++)
			{
				const auto diffPoint = cloudBefore[x] - TransformPoint(cloudAfter[k], rotationMatrix, translationVector, scale);
				float index = multiplier * diffPoint.LengthSquared();
				denominator += std::exp(index);
			}
				
			for (size_t y = 0; y < cloudAfter.size(); y++)
			{
				const auto diffPoint = cloudBefore[x] - TransformPoint(cloudAfter[y], rotationMatrix, translationVector, scale);
				float index = multiplier * diffPoint.LengthSquared();
				float numerator = std::exp(index);
				PMatrix[y * cloudBefore.size() + x] = numerator / denominator;
			}
		}
	}

	void ComputePMatrix2(
		std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared)
	{
		const float multiplier = -0.5f / sigmaSquared;
		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			float denominator = constant;
			for (size_t k = 0; k < cloudAfter.size(); k++)
			{
				const auto diffPoint = cloudBefore[x] - cloudAfter[k];
				float index = multiplier * diffPoint.LengthSquared();
				denominator += std::exp(index);
			}

			for (size_t y = 0; y < cloudAfter.size(); y++)
			{
				const auto diffPoint = cloudBefore[x] - cloudAfter[y];
				float index = multiplier * diffPoint.LengthSquared();
				float numerator = std::exp(index);
				PMatrix[y * cloudBefore.size() + x] = numerator / denominator;
			}
		}
	}

	Probabilities ComputePMatrix3(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		const float& constant,
		const float& sigmaSquared)
	{
		const float multiplier = -0.5f / sigmaSquared;
		Eigen::VectorXf p = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::VectorXf p1 = Eigen::VectorXf::Zero(cloudAfter.size());
		//Eigen::VectorXf p1_max = Eigen::VectorXf::Zero(cloudAfter.size());
		Eigen::VectorXf pt1 = Eigen::VectorXf::Zero(cloudBefore.size());
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudAfter.size(), DIMENSION);
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
			}
		}
		return { p1, pt1, px };
	}

	void MStep(
		const std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix, 
		glm::vec3* translationVector, 
		float* scale, 
		float* sigmaSquared)
	{
		const float Np = std::accumulate(PMatrix.begin(), PMatrix.end(), 0.0f);
		const float InvertedNp = 1.0f / Np;
		printf("Np: %f\n", Np);
		auto EigenBefore = GetMatrix3XFromPointsVector(cloudBefore);
		auto EigenAfter = GetMatrix3XFromPointsVector(cloudAfter);
		const auto EigenPMatrix = GetMatrixXFromPointsVector(PMatrix, cloudAfter.size(), cloudBefore.size());
		Eigen::Vector3f EigenCenterBefore = InvertedNp * EigenBefore * EigenPMatrix.transpose() * Eigen::VectorXf::Ones(cloudAfter.size());
		Eigen::Vector3f EigenCenterAfter = InvertedNp * EigenAfter * EigenPMatrix * Eigen::VectorXf::Ones(cloudBefore.size());

		EigenBefore.colwise() -= EigenCenterBefore;
		EigenAfter.colwise() -= EigenCenterAfter;

		const Eigen::MatrixXf AMatrix = EigenBefore * EigenPMatrix.transpose() * EigenAfter.transpose();
		printf("Amatrix:\n");
		PrintMatrix(AMatrix);
		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVtransposed = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVtransposed;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVtransposed;

		const Eigen::Matrix3f EigenScaleNumerator = AMatrix.transpose() * EigenRotationMatrix;
		//const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		float scaleNumerator = EigenScaleNumerator.trace();
		
		const Eigen::VectorXf PMatrixDiagonalVector = EigenPMatrix * Eigen::VectorXf::Ones(cloudBefore.size());
		const Eigen::VectorXf PMatrixTransposedDiagonalVector = EigenPMatrix.transpose() * Eigen::VectorXf::Ones(cloudAfter.size());
		
		float scaleDenominator = (EigenAfter * PMatrixDiagonalVector.asDiagonal() * EigenAfter.transpose()).trace();

		*scale = scaleNumerator / scaleDenominator;

		const Eigen::Vector3f EigenTranslationVector = EigenCenterBefore - (*scale) * EigenRotationMatrix * EigenCenterAfter;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);

		const float sigmaSubtrahend = (EigenBefore * PMatrixTransposedDiagonalVector.asDiagonal() * EigenBefore.transpose()).trace();

		*sigmaSquared = InvertedNp / DIMENSION * (sigmaSubtrahend - (*scale) * scaleNumerator);

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);
	}

	void MStep2(
		const std::vector<float>& PMatrix,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared)
	{
		const float Np = std::accumulate(PMatrix.begin(), PMatrix.end(), 0.0f);
		const float InvertedNp = 1.0f / Np;
		printf("Np: %f\n", Np);
		auto EigenBefore = GetMatrix3XFromPointsVector(cloudBefore);
		auto EigenAfter = GetMatrix3XFromPointsVector(cloudAfter);
		const auto EigenPMatrix = GetMatrixXFromPointsVector(PMatrix, cloudAfter.size(), cloudBefore.size());
		Eigen::Vector3f EigenCenterBefore = InvertedNp * EigenBefore * EigenPMatrix.transpose() * Eigen::VectorXf::Ones(cloudAfter.size());
		Eigen::Vector3f EigenCenterAfter = InvertedNp * EigenAfter * EigenPMatrix * Eigen::VectorXf::Ones(cloudBefore.size());

		const Eigen::MatrixXf AMatrix = EigenBefore * EigenPMatrix.transpose() * EigenAfter.transpose() - Np * EigenCenterBefore * EigenCenterAfter.transpose();
		printf("Amatrix:\n");
		PrintMatrix(AMatrix);
		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVtransposed = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVtransposed;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVtransposed;

		const Eigen::Matrix3f EigenScaleNumerator = AMatrix.transpose() * EigenRotationMatrix;
		//const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		float scaleNumerator = EigenScaleNumerator.trace();

		const Eigen::VectorXf PMatrixDiagonalVector = EigenPMatrix * Eigen::VectorXf::Ones(cloudBefore.size());
		const Eigen::VectorXf PMatrixTransposedDiagonalVector = EigenPMatrix.transpose() * Eigen::VectorXf::Ones(cloudAfter.size());

		float scaleDenominator = (EigenAfter * PMatrixDiagonalVector.asDiagonal() * EigenAfter.transpose()).trace();

		*scale = scaleNumerator / scaleDenominator;

		const Eigen::Vector3f EigenTranslationVector = EigenCenterBefore - (*scale) * EigenRotationMatrix * EigenCenterAfter;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);

		const float sigmaSubtrahend = (EigenBefore * PMatrixTransposedDiagonalVector.asDiagonal() * EigenBefore.transpose()).trace();

		*sigmaSquared = InvertedNp / DIMENSION * (sigmaSubtrahend - (*scale) * scaleNumerator);

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);
	}

	void MStep3(
		const Probabilities& probabilities,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared)
	{
		const float Np = probabilities.p1.sum();
		const float InvertedNp = 1.0f / Np;
		printf("Np: %f\n", Np);
		auto EigenBeforeT = GetMatrix3XFromPointsVector(cloudBefore);
		auto EigenAfterT = GetMatrix3XFromPointsVector(cloudAfter);
		Eigen::Vector3f EigenCenterBefore = InvertedNp * EigenBeforeT * probabilities.pt1;
		Eigen::Vector3f EigenCenterAfter = InvertedNp * EigenAfterT * probabilities.p1;

		//const Eigen::MatrixXf AMatrix = (probabilities.px.transpose() * EigenAfterT.transpose());
		const Eigen::MatrixXf AMatrix = (EigenAfterT * probabilities.px).transpose() - Np * (EigenCenterBefore * EigenCenterAfter.transpose());
		printf("AmatrixMSTEP3:\n");
		PrintMatrix(AMatrix);
		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVT = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVT;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1.0f, 1.0f, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVT;

		//const Eigen::Matrix3f EigenScaleNumerator = AMatrix.transpose() * EigenRotationMatrix;
		const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		const float scaleNumerator = EigenScaleNumerator.trace();

		//const Eigen::VectorXf PMatrixDiagonalVector = EigenPMatrix * Eigen::VectorXf::Ones(cloudBefore.size());
		//const Eigen::VectorXf PMatrixTransposedDiagonalVector = EigenPMatrix.transpose() * Eigen::VectorXf::Ones(cloudAfter.size());

		const float scaleDenominator = (EigenAfterT.transpose().array().pow(2) * probabilities.p1.replicate(1, DIMENSION).array()).sum() 
			- Np * EigenCenterAfter.transpose() * EigenCenterAfter;

		*scale = scaleNumerator / scaleDenominator;
		//float scale_tmp = scaleNumerator / scaleDenominator;

		const Eigen::Vector3f EigenTranslationVector = EigenCenterBefore - (*scale) * EigenRotationMatrix * EigenCenterAfter;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);
		//auto translationVector_tmp = ConvertTranslationVector(EigenTranslationVector);

		const float sigmaSubtrahend = (EigenBeforeT.transpose().array().pow(2) * probabilities.pt1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterBefore.transpose() * EigenCenterBefore;

		*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / DIMENSION;
		//float sigmaSquared_tmp = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / DIMENSION;

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);
		//auto rotationMatrix_tmp = ConvertRotationMatrix(EigenRotationMatrix);

		//printf("MStep3, sigmaSquared: %f, scale: %f\nTransformation Matrix:\n", sigmaSquared_tmp, *scale);
		//PrintMatrix(ConvertToTransformationMatrix(*scale * *rotationMatrix, translationVector));
	}
}