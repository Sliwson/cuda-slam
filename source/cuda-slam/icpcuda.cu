#include "icpcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"

using namespace Common;
using namespace CUDACommon;

namespace
{
	glm::mat4 CudaICP(const GpuCloud& before, const GpuCloud& after)
	{
		const int maxIterations = 60;
		const float TEST_EPS = 1e-5;
		float previousError = std::numeric_limits<float>::max();

		int iterations = 0;
		glm::mat4 transformationMatrix(1.0f);
		glm::mat4 previousTransformationMatrix = transformationMatrix;


		//do not change before vector - copy it for calculations
		const int beforeSize = before.size();
		const int afterSize = after.size();

		GpuCloud workingBefore(beforeSize);
		GpuCloud alignBefore(beforeSize);
		GpuCloud alignAfter(beforeSize);

		thrust::device_vector<int> indices(beforeSize);
		thrust::copy(thrust::device, before.begin(), before.end(), workingBefore.begin());

		//allocate memory for cuBLAS
		CudaSvdParams params(beforeSize, beforeSize, 3, 3);

		while (iterations < maxIterations)
		{
			GetCorrespondingPoints(indices, workingBefore, after);

			transformationMatrix = LeastSquaresSVD(indices, workingBefore, after, alignBefore, alignAfter, params) * transformationMatrix;

			TransformCloud(before, workingBefore, transformationMatrix);
			float error = GetMeanSquaredError(indices, workingBefore, after);
			printf("Iteration: %d, error: %f\n", iterations, error);
			if (error < TEST_EPS)
				break;

			if (error > previousError)
			{
				printf("Error has increased, aborting\n");
				transformationMatrix = previousTransformationMatrix;
				break;
			}

			previousTransformationMatrix = transformationMatrix;
			previousError = error;
			iterations++;
		}

		params.Free();
		return transformationMatrix;
	}
}
