#include "fgt.h"

using namespace Common;

namespace FastGaussTransform
{
	struct FGT_Model
	{
		// The K center points of the training set (d x K)
		std::vector<Point_f> xc;
		// Polynomial coefficient (pd x K), where pd = nchoosek(p + d - 1 , d)
		Eigen::MatrixXf Ak;
	};

	//idmax -> zwraca indeks najwiekszego elementu w tablicy idmax(tablica, rozmiar)
	//ddist -> zwraca sume kwadratow roznic elementow dwoch tablic ddist(tablica1,tablica2,wymiar)

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

		Eigen::MatrixXf A_k = Eigen::ArrayXf::Zero(pd, K_param);

		KCenter(cloud, K_param, &xc, &indx);
		ComputeC_k(p_param, &C_k);
		ComputeA_k(cloud, weights, xc, C_k, indx, sigma, K_param, p_param, pd, &A_k);

		return { xc, A_k };
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
			const auto furthest_point = std::max_element(dist_C.begin(), dist_C.end());
			center_ind = std::distance(dist_C.begin(), furthest_point);
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