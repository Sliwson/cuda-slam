#pragma once
#include <utility>
#include <tuple>
#include "common.h"

namespace FastGaussTransform
{
	struct FGT_Model;

	FGT_Model ComputeFGTModel(
		const std::vector<Common::Point_f>& cloud,
		const std::vector<float>& weights,
		const float& sigma,
		const int& K_param,
		const int& p_param);

	std::vector<float> ComputeFGTPredict(
		const std::vector<Common::Point_f>& cloud,
		const FGT_Model& fgt_model,
		const float& sigma,
		const float& e_param,
		const int& K_param,
		const int& p_param);
}