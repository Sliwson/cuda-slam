#pragma once

#include "_common.h"
#include "common.h"

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

#include <cublas_v2.h>
#include <cusolverDn.h>

#define USE_CORRESPONDENCES_KERNEL

using namespace Common;

void CudaTest();
void CudaTest2();
