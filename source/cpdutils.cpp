#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include "cpdutils.h"
#include "fgt.h"
#include "fgt_model.h"

using namespace CoherentPointDrift;
using namespace Common;
using namespace FastGaussTransform;

namespace CPDutils
{
	Eigen::VectorXf CalculatePt1(const std::vector<float>& Kt1, const float& ndi);
	std::vector<float> CalculateWeightsForPX(
		const std::vector<Point_f>& cloud,
		const std::vector<float>& invDenomP,
		const int& row);

	Probabilities ComputePMatrixWithFGT(
		const std::vector<Point_f>& cloudTransformed,
		const std::vector<Point_f>& cloudAfter,		
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit)
	{
		const int N = cloudAfter.size();
		const int M = cloudTransformed.size();

		const float hsigma = std::sqrt(2.0f * sigmaSquared);

		//FGT parameters
		float e_param = 9.0f; //Ratio of far field (default e = 10)
		int K_param = std::round(std::min({ (float)N, (float)M, 50.0f + sigmaSquaredInit / sigmaSquared }));
		int p_param = 6; //Order of truncation (default p = 8)

		FGT_Model fgt_model;

		//compute pt1 and denom
		fgt_model = ComputeFGTModel(cloudTransformed, std::vector<float>(M, 1.0f), hsigma, K_param, p_param);
		auto Kt1 = ComputeFGTPredict(cloudAfter, fgt_model, hsigma, e_param, K_param, p_param);

		const float ndi = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * M) / ((1 - weight) * N);

		//transform Kt1 to 1./denomP
		auto invDenomP = std::vector<float>(Kt1.size());
		std::transform(Kt1.begin(), Kt1.end(), invDenomP.begin(), [&ndi](const float& p) {return 1.0f / (p + ndi); });

		Eigen::VectorXf pt1 = CalculatePt1(invDenomP, ndi);

		//compute P1
		fgt_model = ComputeFGTModel(cloudAfter, invDenomP, hsigma, K_param, p_param);
		auto P1_vector = ComputeFGTPredict(cloudTransformed, fgt_model, hsigma, e_param, K_param, p_param);
		Eigen::VectorXf p1 = GetVectorXFromPointsVector(P1_vector);

		//compute PX
		Eigen::MatrixXf px = Eigen::MatrixXf::Zero(cloudTransformed.size(), DIMENSION);
		for (int i = 0; i < DIMENSION; i++)
		{
			fgt_model = ComputeFGTModel(cloudAfter, CalculateWeightsForPX(cloudAfter, invDenomP, i), hsigma, K_param, p_param);
			auto result_vector = ComputeFGTPredict(cloudTransformed, fgt_model, hsigma, e_param, K_param, p_param);
			Eigen::VectorXf result_eigen = GetVectorXFromPointsVector(result_vector);
			px.col(i) = result_eigen;
		}

		//calculate error
		std::transform(Kt1.begin(), Kt1.end(), Kt1.begin(), [&ndi](const float& p) {return std::log(p + ndi); });
		float error = -std::accumulate(Kt1.begin(), Kt1.end(), 0.0f);
		error += DIMENSION * N * std::log(sigmaSquared) / 2.0f;

		return { p1, pt1, px, error };
	}

	Eigen::VectorXf CalculatePt1(const std::vector<float>& invDenomP, const float& ndi)
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
}
