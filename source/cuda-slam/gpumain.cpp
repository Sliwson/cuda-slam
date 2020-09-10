#include "cpdcuda.cuh"
#include "nicpcuda.cuh"
#include "icpcuda.cuh"

#include "mainwrapper.h"

using namespace Common;

#define TEST

namespace {
	std::pair<glm::mat3, glm::vec3> GetGpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations, float* error)
	{
		const auto maxIterations = configuration.MaxIterations.has_value() ? configuration.MaxIterations.value() : -1;

		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				return GetCudaIcpTransformationMatrix(
					before, after, 
					configuration.ConvergenceEpsilon, maxIterations,
					iterations, error);
			case ComputationMethod::NoniterativeIcp:
				return GetCudaNicpTransformationMatrix(
					before, after,
					configuration.ConvergenceEpsilon, configuration.NicpIterations, configuration.NicpBatchSize, configuration.ApproximationType, configuration.NicpSubcloudSize,
					iterations, error);
			case ComputationMethod::Cpd:
				return GetCudaCpdTransformationMatrix(before, after,
					configuration.ConvergenceEpsilon, configuration.CpdWeight, configuration.CpdConstScale, maxIterations, configuration.CpdTolerance, configuration.ApproximationType,
					iterations, error);
			default:
				assert(false); //unknown method
				return GetCudaIcpTransformationMatrix(
					before, after, 
					configuration.ConvergenceEpsilon, maxIterations,
					iterations, error);
		}
	}

	int RunGpuTests()
	{ 
		srand(Tests::RANDOM_SEED);

		const auto methods = { ComputationMethod::Icp, ComputationMethod::NoniterativeIcp, ComputationMethod::Cpd };
		Tests::RunTestSet(GetConvergenceTestSet, GetGpuSlamResult, "convergence-gpu", methods);
		return 0;
	}
}

int main(int argc, char** argv)
{
#ifdef TEST
	return RunGpuTests();
#else
	return Main(argc, argv, "Gpu Slam", GetGpuSlamResult);
#endif
}
