/*
	This file contains slightly modified and refactored code created by Sebastien Paris.

	Copyright (c) 2011, Sebastien Paris
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice, this
	  list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above copyright notice,
	  this list of conditions and the following disclaimer in the documentation
	  and/or other materials provided with the distribution
	* Neither the name of LIS/DYNI, University of Toulon, UMR 7020 nor the names of its
	  contributors may be used to endorse or promote products derived from this
	  software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "fgt.h"
#include "fgt_model.h"

using namespace Common;

namespace FastGaussTransform
{
	void KCenter(
		const std::vector<Point_f>& cloud,
		const int& K_param,
		std::vector<Point_f>* xc,
		std::vector<int>* indx);

	void ComputeC_k(const int& p_param, std::vector<float>* C_k);

	void ComputeA_k(
		const std::vector<Point_f>& cloud,
		const std::vector<float>& weights,
		const std::vector<Point_f>& xc,
		const std::vector<float>& C_k,
		const std::vector<int>& indx,
		const float& sigma,
		const int& K_param,
		const int& p_param,
		const int& pd,
		Eigen::MatrixXf* A_k);

	int nchoosek(const int& n, int k);

	FGT_Model ComputeFGTModel(
		const std::vector<Point_f>& cloud,
		const std::vector<float>& weights,
		const float& sigma,
		const int& K_param,
		const int& p_param)
	{
		const int Nx = cloud.size();
		const int pd = nchoosek(p_param + DIMENSION - 1, DIMENSION);

		auto xc = std::vector<Point_f>(K_param);
		auto indx = std::vector<int>(Nx);
		auto C_k = std::vector<float>(pd);

		Eigen::MatrixXf A_k = Eigen::ArrayXXf::Zero(pd, K_param);

		KCenter(cloud, K_param, &xc, &indx);
		ComputeC_k(p_param, &C_k);
		ComputeA_k(cloud, weights, xc, C_k, indx, sigma, K_param, p_param, pd, &A_k);

		return { xc, A_k };
	}

	std::vector<float> ComputeFGTPredict(
		const std::vector<Common::Point_f>& cloud,
		const FGT_Model& fgt_model,
		const float& sigma,
		const float& e_param,
		const int& K_param,
		const int& p_param)
	{
		const int Ny = cloud.size();
		const int pd = fgt_model.Ak.rows();
		const float invertedSigma = 1.0f / sigma;
		Point_f dy = Point_f::Zero();
		int k, t, tail, head, ind;

		auto prods = std::vector<float>(pd);
		auto heads = std::vector<int>(DIMENSION + 1);
		auto v = std::vector<float>(Ny);

		heads[DIMENSION] = std::numeric_limits<int>::max();

		for (int m = 0; m < Ny; m++)
		{
			float cell_sum = 0.0;
			for (int kn = 0; kn < K_param; kn++)
			{
				ind = kn * pd;
				for (int i = 0; i < DIMENSION; i++)
				{
					heads[i] = 0;
				}

				dy = cloud[m] - fgt_model.xc[kn];
				dy *= invertedSigma;
				const float sum = dy.LengthSquared();

				if (sum > e_param) continue; //skip to next kn

				prods[0] = std::exp(-sum);

				for (k = 1, t = 1, tail = 1; k < p_param; k++, tail = t)
				{
					for (int i = 0; i < DIMENSION; i++)
					{
						head = heads[i];
						heads[i] = t;
						const float val = dy[i];
						for (int j = head; j < tail; j++, t++)
						{
							prods[t] = val * prods[j];
						}
					}
				}

				for (int i = 0; i < pd; i++)
				{
					cell_sum += fgt_model.Ak.data()[i + ind] * prods[i];
				}
			}
			v[m] = cell_sum;
		}
		return v;
	}

	void KCenter(
		const std::vector<Point_f>& cloud,
		const int& K_param,
		std::vector<Point_f>* xc,
		std::vector<int>* indx)
	{
		const int Nx = cloud.size();
		//auto indxc = std::vector<int>(K_param);
		auto xboxsz = std::vector<int>(K_param);
		auto dist_C = std::vector<float>(Nx);
		int center_ind = 1;
		//int indxc_index = 0;

		//randomly pick one node as the first center.
		//srand((unsigned)time(NULL));
		//ind = rand() % Nx;

		//indxc[indxc_index++] = center_ind;

		for (int i = 0; i < Nx; i++)
		{
			const Point_f diff = cloud[i] - cloud[center_ind];
			dist_C[i] = diff.LengthSquared();
			(*indx)[i] = 0;
		}

		for (int i = 1; i < K_param; i++)
		{
			const auto furthestPoint = std::max_element(dist_C.begin(), dist_C.end());
			center_ind = std::distance(dist_C.begin(), furthestPoint);
			//indxc[indxc_index++] = center_ind;
			for (int j = 0; j < Nx; j++)
			{
				const Point_f diff = cloud[j] - cloud[center_ind];
				float dist = diff.LengthSquared();

				if (dist < dist_C[j])
				{
					dist_C[j] = dist;
					(*indx)[j] = i;
				}
			}
		}

		for (int i = 0; i < K_param; i++)
		{
			xboxsz[i] = 0;
			(*xc)[i] = Point_f::Zero();
		}

		for (int i = 0; i < Nx; i++)
		{
			xboxsz[(*indx)[i]]++;
			(*xc)[(*indx)[i]] += cloud[i];
		}

		for (int i = 0; i < K_param; i++)
		{
			(*xc)[i] *= 1.0f / (float)xboxsz[i];
		}
	}

	void ComputeC_k(const int& p_param, std::vector<float>* C_k)
	{
		auto cinds = std::vector<int>(C_k->size());
		auto heads = std::vector<int>(DIMENSION + 1);

		for (int i = 0; i < DIMENSION; i++)
		{
			heads[i] = 0;
		}
		heads[DIMENSION] = std::numeric_limits<int>::max();

		cinds[0] = 0;
		(*C_k)[0] = 1.0f;

		int k, t, tail, head;
		for (k = 1, t = 1, tail = 1; k < p_param; k++, tail = t)
		{
			for (int i = 0; i < DIMENSION; i++)
			{
				head = heads[i];
				heads[i] = t;

				for (int j = head; j < tail; j++, t++)
				{
					cinds[t] = (j < heads[i + 1]) ? cinds[j] + 1 : 1;
					(*C_k)[t] = 2.0 * (*C_k)[j];
					(*C_k)[t] /= (double)cinds[t];
				}
			}
		}
	}

	void ComputeA_k(
		const std::vector<Point_f>& cloud,
		const std::vector<float>& weights,
		const std::vector<Point_f>& xc,
		const std::vector<float>& C_k,
		const std::vector<int>& indx,
		const float& sigma,
		const int& K_param,
		const int& p_param,
		const int& pd,
		Eigen::MatrixXf* A_k)
	{
		const int Nx = cloud.size();
		const float invertedSigma = 1.0f / sigma;
		Point_f dx = Point_f::Zero();
		int k, t, tail, head;

		auto prods = std::vector<float>(pd);
		auto heads = std::vector<int>(DIMENSION + 1);

		heads[DIMENSION] = std::numeric_limits<int>::max();

		for (int n = 0; n < Nx; n++)
		{
			dx = cloud[n] - xc[indx[n]];
			dx *= invertedSigma;

			prods[0] = std::exp(-dx.LengthSquared());

			for (int i = 0; i < DIMENSION; i++)
			{
				heads[i] = 0;
			}

			for (k = 1, t = 1, tail = 1; k < p_param; k++, tail = t)
			{
				for (int i = 0; i < DIMENSION; i++)
				{
					head = heads[i];
					heads[i] = t;
					float val = dx[i];

					for (int j = head; j < tail; j++, t++)
					{
						prods[t] = val * prods[j];
					}
				}
			}

			for (int i = 0; i < pd; i++)
			{
				(*A_k)(i, indx[n]) += weights[n] * prods[i];
			}
		}

		for (int k = 0; k < K_param; k++)
		{
			for (int i = 0; i < pd; i++)
			{
				(*A_k)(i, k) *= C_k[i];
			}
		}
	}

	int nchoosek(const int& n, int k)
	{
		int n_k = n - k;
		int nchsk = 1;
		if (k < n_k)
		{
			k = n_k;
			n_k = n - k;
		}

		for (int i = 1; i <= n_k; i++)
		{
			nchsk *= (++k);
			nchsk /= i;
		}

		return nchsk;
	}
}
