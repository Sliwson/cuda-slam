#pragma once

#include "cuda.cuh"
#include "cudaprobabilities.h"

namespace MStepParams
{
	struct CUDAMStepParams
	{
		CUDAMStepParams(const int& beforeLength, const int& afterLength, const CUDAProbabilities::Probabilities& probabilities)
		{
			cudaMalloc((void**)&beforeT, beforeLength * 3 * sizeof(float));
			cudaMalloc((void**)&afterT, afterLength * 3 * sizeof(float));
			cudaMalloc((void**)&centerBefore, 3 * sizeof(float));
			cudaMalloc((void**)&centerAfter, 3 * sizeof(float));
			cudaMalloc((void**)&AMatrix, 9 * sizeof(float));
			cudaMalloc((void**)&px, 3 * afterLength * sizeof(float));
			cudaMalloc((void**)&afterTxPX, 9 * sizeof(float));
			cudaMalloc((void**)&centerBeforexCenterAfter, 9 * sizeof(float));

			p1 = thrust::raw_pointer_cast(probabilities.p1.data());
			pt1 = thrust::raw_pointer_cast(probabilities.pt1.data());

			cublasCreate(&multiplyHandle);

			cudaMalloc((void**)&devInfo, sizeof(int));
			cudaMalloc((void**)&S, 3 * sizeof(float));
			cudaMalloc((void**)&VT, 9 * sizeof(float));
			cudaMalloc((void**)&U, 9 * sizeof(float));
			cusolverDnCreate(&solverHandle);

			cusolverDnDgesvd_bufferSize(solverHandle, 3, 3, &workSize);
			cudaMalloc((void**)&work, workSize * sizeof(float));
		}

		void Free()
		{
			cudaFree(beforeT);
			cudaFree(afterT);
			cudaFree(centerBefore);
			cudaFree(centerAfter);
			cudaFree(AMatrix);
			cudaFree(px);
			cudaFree(afterTxPX);
			cudaFree(centerBeforexCenterAfter);

			cublasDestroy(multiplyHandle);

			cudaFree(work);
			cudaFree(devInfo);
			cudaFree(S);
			cudaFree(VT);
			cudaFree(U);
			cusolverDnDestroy(solverHandle);
		}

		float* beforeT = nullptr;
		float* afterT = nullptr;
		float* centerBefore = nullptr;
		float* centerAfter = nullptr;
		float* AMatrix = nullptr;
		const float* p1 = nullptr;
		const float* pt1 = nullptr;
		float* px = nullptr;
		float* afterTxPX = nullptr;
		float* centerBeforexCenterAfter = nullptr;

		cublasHandle_t multiplyHandle = nullptr;

		//for svd
		float* S = nullptr;
		float* VT = nullptr;
		float* U = nullptr;
		float* work = nullptr;
		int* devInfo = nullptr;
		int workSize = 0;
		cusolverDnHandle_t solverHandle = nullptr;
	};
}
