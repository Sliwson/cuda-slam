#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "coherentpointdrift.h"
#include "fgt.h"
#include "fgt_model.h"

using namespace Common;
using namespace FastGaussTransform;

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

	float CalculateSigmaSquared(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);
	//Probabilities ComputePMatrix(
	//	const std::vector<Point_f>& cloudBefore,
	//	const std::vector<Point_f>& cloudTransformed,
	//	const float& constant,
	//	const float& sigmaSquared);
	Probabilities ComputePMatrixFast(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const int& fgt);
	Probabilities ComputePMatrixWithFGT(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit);
	Eigen::VectorXf CalculatePt1(const std::vector<float>& Kt1, const float& ndi);
	std::vector<float> CalculateWeightsForPX(
		const std::vector<Point_f>& cloud,
		const std::vector<float>& invDenomP,
		const int& row);
	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate = false,
		float truncate = -1.0f);
	void MStep(
		const Probabilities& probabilities,
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		bool const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

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
		int fgt)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		float scale = 1.0f;
		float sigmaSquared = CalculateSigmaSquared(cloudBefore, cloudAfter);
		float sigmaSquared_init = sigmaSquared;
		//TODO: add check for weight=1
		weight = std::clamp(weight, 0.0f, 1.0f);

		if (weight == 0.0f)
		{
			//weight = 10 * std::numeric_limits<float>::min();
			weight = 1e-6f;
		}

		if (weight == 1.0f)
		{
			//outliers = 10 * std::numeric_limits<float>::min();
			weight = 1.0f - 1e-6f;
		}
		const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudAfter.size()) / ((1 - weight) * cloudBefore.size());
		float ntol = tolerance + 10.0f;
		float l = 0.0f;
		Probabilities probabilities;
		std::vector<Point_f> transformedCloud = cloudAfter;
		//EM optimization
		while (*iterations < maxIterations && ntol > tolerance && sigmaSquared > eps)
		{
			//E-step
			if (fgt == 0)
			{
				probabilities = ComputePMatrix(cloudBefore, transformedCloud, constant, sigmaSquared);
			}
			else
			{
				probabilities = ComputePMatrixFast(cloudBefore, transformedCloud, constant, weight, &sigmaSquared, sigmaSquared_init, fgt);
			}

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
		if (probabilities.correspondence.size() == transformedCloud.size())
			*error = GetMeanSquaredError(cloudBefore, transformedCloud, probabilities.correspondence);
		return std::make_pair(scale * rotationMatrix, translationVector);
	}

	void TestFGT()
	{
		auto cloudY = std::vector<Point_f>();

		cloudY.push_back(Point_f(0.309336386192094,
			0.707084625951691,
			0.298113288774646));
		cloudY.push_back(Point_f(0.927420719822510,
			1.13723783523452,
			-1.06097318210616));
		cloudY.push_back(Point_f(-0.193269258310719,
			-1.04186474555901,
			-1.17995754544710));
		cloudY.push_back(Point_f(0.0171965470714722,
			-1.24309161063860,
			1.08001918998961));
		cloudY.push_back(Point_f(-0.835916847502900,
			0.492095628437480,
			-0.372592706818986));
		cloudY.push_back(Point_f(1.09689261405179,
			-1.96878196118039,
			-1.81027465424799));
		cloudY.push_back(Point_f(-1.61339867481782,
			-1.44189376201321,
			-1.12133640172003));
		cloudY.push_back(Point_f(1.05062755717162,
			-0.409510045836082,
			-1.59301882701997));
		cloudY.push_back(Point_f(0.968175497182022,
			1.51454386177120,
			0.181131970076453));
		cloudY.push_back(Point_f(-0.331322753540273,
			0.192331269482018,
			-0.516441645483671));

		auto cloudX = std::vector<Point_f>();

		cloudX.push_back(Point_f(0.947451590180286,
			0.340669264759082,
			0.722509917468891));
		cloudX.push_back(Point_f(0.192703847861963,
			-0.307508917315778,
			1.16455724773272));
		cloudX.push_back(Point_f(-0.411094460953666,
			-1.72914384589898,
			0.685663942880554));
		cloudX.push_back(Point_f(-0.659942503374938,
			-2.18039243158201,
			1.68767712464063));
		cloudX.push_back(Point_f(-0.844198688901197,
			-0.631073992000756,
			0.152910784474644));
		cloudX.push_back(Point_f(1.58641581660387,
			2.09567581741100,
			-0.390064674234967));
		cloudX.push_back(Point_f(-0.454747083593913,
			-1.39444619920813,
			-1.14735053624287));

		auto weights = std::vector(10, 1.0f);

		weights[1] = 1.0f;

		float sigma = 0.3f;
		int k = 5;
		int p = 2;
		int e = 10.0f;


		Probabilities prob = ComputePMatrixFast(cloudX, cloudY,0.1, 0.3, &sigma, 0.5, 1);

		std::cout << "Vector P1" << std::endl;
		std::cout << prob.p1 << std::endl;

		std::cout << "Vector PT1" << std::endl;
		std::cout << prob.pt1 << std::endl;

		std::cout << "Matrix PX" << std::endl;
		std::cout << prob.px << std::endl;

		std::cout << "error" << std::endl;
		std::cout << prob.error << std::endl;
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

	/*Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& constant,
		const float& sigmaSquared)
	{
		const float multiplier = -0.5f / sigmaSquared;
		Eigen::VectorXf p = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf p1 = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf p1_max = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf pt1 = Eigen::VectorXf::Zero(cloudBefore.size());
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudTransformed.size(), DIMENSION);
		std::vector<int> correspondece = std::vector<int>(cloudTransformed.size());
		float error = 0.0;
		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			float denominator = 0.0;
			for (size_t k = 0; k < cloudTransformed.size(); k++)
			{
				const auto diffPoint = cloudBefore[x] - cloudTransformed[k];
				float index = multiplier * diffPoint.LengthSquared();
				float value = std::exp(index);
				p(k) = value;
				denominator += value;
			}
			denominator += constant;
			pt1(x) = 1.0f - constant / denominator;
			for (size_t k = 0; k < cloudTransformed.size(); k++)
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
		error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2.0f;

		return { p1, pt1, px, error, correspondece };
	}*/

	Probabilities ComputePMatrixFast(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const int& fgt)
	{
		if (fgt == 1)
		{
			if (*sigmaSquared < 0.05)
				*sigmaSquared = 0.05;
			return ComputePMatrixWithFGT(cloudBefore, cloudTransformed, weight, *sigmaSquared, sigmaSquaredInit);
		}
		if (fgt == 2)
		{
			if (*sigmaSquared > 0.015 * sigmaSquaredInit)
				return ComputePMatrixWithFGT(cloudBefore, cloudTransformed, weight, *sigmaSquared, sigmaSquaredInit);
			else
				return ComputePMatrix(cloudBefore, cloudTransformed, constant, *sigmaSquared, true, 1e-3f);
		}
		return Probabilities();
	}

	Probabilities ComputePMatrixWithFGT(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit)
	{
		const int N = cloudBefore.size();
		const int M = cloudTransformed.size();

		const float hsigma = std::sqrt(2.0f * sigmaSquared);	

		//FGT parameters
		float e_param = 9.0f; //Ratio of far field (default e = 10)
		int K_param = std::round(std::min({ (float)N, (float)M, 50.0f + sigmaSquaredInit / sigmaSquared }));
		int p_param = 6; //Order of truncation (default p = 8)

		FGT_Model fgt_model;

		//compute pt1 and denom
		fgt_model = ComputeFGTModel(cloudTransformed, std::vector<float>(M, 1.0f), hsigma, K_param, p_param);
		auto Kt1 = ComputeFGTPredict(cloudBefore, fgt_model, hsigma, e_param, K_param, p_param);

		const float ndi = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * M) / ((1 - weight) * N);

		//transform Kt1 to 1./denomP
		auto invDenomP = std::vector<float>(Kt1.size());
		std::transform(Kt1.begin(), Kt1.end(), invDenomP.begin(), [&ndi](const float& p) {return 1.0f / (p + ndi); });

		Eigen::VectorXf pt1 = CalculatePt1(invDenomP, ndi);

		//compute P1
		fgt_model = ComputeFGTModel(cloudBefore, invDenomP, hsigma, K_param, p_param);
		auto P1_vector = ComputeFGTPredict(cloudTransformed, fgt_model, hsigma, e_param, K_param, p_param);
		Eigen::VectorXf p1 = GetVextorXFromPointsVector(P1_vector);

		//compute PX
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudTransformed.size(), DIMENSION);
		for (int i = 0; i < DIMENSION; i++)
		{
			fgt_model = ComputeFGTModel(cloudBefore, CalculateWeightsForPX(cloudBefore, invDenomP, i), hsigma, K_param, p_param);
			auto result_vector = ComputeFGTPredict(cloudTransformed, fgt_model, hsigma, e_param, K_param, p_param);
			Eigen::VectorXf result_eigen = GetVextorXFromPointsVector(result_vector);
			px.col(i) = result_eigen;
		}

		//calculate error
		std::transform(Kt1.begin(), Kt1.end(), Kt1.begin(), [&ndi](const float& p) {return std::log(p + ndi); });
		float error = -std::accumulate(Kt1.begin(), Kt1.end(), 0.0f);
		error += DIMENSION * N * std::log(sigmaSquared) / 2.0f;

		//correspondences
		//TODO: maybe delete it
		std::vector<int> correspondece = std::vector<int>();

		return { p1, pt1, px, error, correspondece };
	}

	Eigen::VectorXf CalculatePt1(const std::vector<float>& invDenomP,const float& ndi)
	{
		const int N = invDenomP.size();
		Eigen::VectorXf Pt1 = Eigen::VectorXf::Zero(N);
		for (int i = 0; i < N; i++)
		{
			Pt1(i) = 1.0f - ndi * invDenomP[i];
		}
		return Pt1;
	}

	std::vector<float> CalculateWeightsForPX(const std::vector<Point_f>& cloud, const std::vector<float>& invDenomP, const int& row)
	{
		const int N = cloud.size();
		auto result = std::vector<float>(N);
		for (int i = 0; i < N; i++)
		{
			result[i] = cloud[i][row] * invDenomP[i];
		}
		return result;
	}

	Probabilities ComputePMatrix(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudTransformed,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate)
	{
		const float multiplier = -0.5f / sigmaSquared;
		Eigen::VectorXf p = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf p1 = Eigen::VectorXf::Zero(cloudTransformed.size());
		//Eigen::VectorXf p1_max = Eigen::VectorXf::Zero(cloudTransformed.size());
		Eigen::VectorXf pt1 = Eigen::VectorXf::Zero(cloudBefore.size());
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudTransformed.size(), DIMENSION);
		//TODO: maybe delete correspondence
		std::vector<int> correspondece = std::vector<int>();
		float error = 0.0;
		if (doTruncate)
			truncate = std::log(truncate);

		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			float denominator = 0.0;
			for (size_t k = 0; k < cloudTransformed.size(); k++)
			{
				const auto diffPoint = cloudBefore[x] - cloudTransformed[k];
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
					px.row(k) += ConvertToEigenVector(cloudBefore[x]) * value;
				}
			}
			error -= std::log(denominator);
		}
		error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2.0f;

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