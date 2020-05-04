#pragma once
#include "_common.h"
#include <Eigen/Dense>

namespace FastGaussTransform
{
	struct FGT_Model
	{
		// The K center points of the training set (d x K)
		std::vector<Common::Point_f> xc;
		// Polynomial coefficient (pd x K), where pd = nchoosek(p + d - 1 , d)
		Eigen::MatrixXf Ak;

		//FGT_Model(std::vector<Common::Point_f> xc, Eigen::MatrixXf A_k)
		//{
		//	this->xc = xc;
		//	this->Ak = A_k;
		//}
	};
}