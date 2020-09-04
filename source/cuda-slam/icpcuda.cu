#include "icpcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"

using namespace Common;
using namespace CUDACommon;

glm::mat4 CudaICP(const GpuCloud& before, const GpuCloud& after, int maxIterations, float eps, int* iterations, float* error)
{
	float previousError = std::numeric_limits<float>::max();

	*iterations = 0;
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

	while (maxIterations == -1 || *iterations < maxIterations)
	{
		GetCorrespondingPoints(indices, workingBefore, after);

		transformationMatrix = LeastSquaresSVD(indices, workingBefore, after, alignBefore, alignAfter, params) * transformationMatrix;

		TransformCloud(before, workingBefore, transformationMatrix);
		*error = GetMeanSquaredError(indices, workingBefore, after);
		printf("Iteration: %d, error: %f\n", *iterations, *error);
		if (*error < eps)
			break;

		if (*error > previousError)
		{
			printf("Error has increased, aborting\n");
			transformationMatrix = previousTransformationMatrix;
			*error = previousError;
			break;
		}

		previousTransformationMatrix = transformationMatrix;
		previousError = *error;
		(*iterations)++;
	}

	params.Free();
	return transformationMatrix;
}

std::pair<glm::mat3, glm::vec3> GetCudaIcpTransformationMatrix(
	const std::vector<Common::Point_f>& cloudBefore,
	const std::vector<Common::Point_f>& cloudAfter,
	float eps,
	int maxIterations,
	int* iterations,
	float* error)
{
	GpuCloud before(cloudBefore.size());
	GpuCloud after(cloudAfter.size());

	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(before.data()), cloudBefore.data(), cloudBefore.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(thrust::raw_pointer_cast(after.data()), cloudAfter.data(), cloudAfter.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

	auto matrix = CudaICP(before, after, maxIterations, eps, iterations, error);
	return Common::ConvertToRotationTranslationPair(matrix);
}

