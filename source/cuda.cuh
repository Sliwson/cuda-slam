#pragma once

#include "_common.h"
#include "common.h"

#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

#include <chrono>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_operation.hpp>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#define USE_CORRESPONDENCES_KERNEL

using namespace Common;

struct CudaSvdParams;

namespace CUDACommon
{
	typedef thrust::device_vector<glm::vec3> GpuCloud;
	typedef thrust::device_vector<int> IndexIterator;
	typedef thrust::permutation_iterator<GpuCloud, IndexIterator> Permutation;	

	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec);
	std::vector<Point_f> ThrustToCommonVector(const GpuCloud& vec);
	glm::vec3 CalculateCentroid(const GpuCloud& vec);
	void TransformCloud(const GpuCloud& vec, GpuCloud& out, const glm::mat4& transform);
	__device__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second);
	float GetMeanSquaredError(const IndexIterator& permutation, const GpuCloud& before, const GpuCloud& after);
	void GetAlignedCloud(const GpuCloud& source, GpuCloud& target);
	void CuBlasMultiply(float* A, float* B, float* C, int size, CudaSvdParams& params);
	glm::mat3 CreateGlmMatrix(float* squareMatrix);
	glm::mat4 LeastSquaresSVD(const IndexIterator& permutation, const GpuCloud& before, const GpuCloud& after, GpuCloud& alignBefore, GpuCloud& alignAfter, CudaSvdParams params);
}
