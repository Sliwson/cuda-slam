#include "cpdcuda.cuh"
#include "nicpcuda.cuh"
#include "icpcuda.cuh"

#include "mainwrapper.h"

using namespace Common;

#define TEST

namespace {
	constexpr auto eps = 1e-5f;

	// TODO: move to configuration class
	constexpr auto nicpIterDefault = 20;
	constexpr auto nicpSubcloudDefault = 1000;
	constexpr auto nicpBatchSize = 16;
	constexpr auto cpdConstScale = true;

	std::pair<glm::mat3, glm::vec3> GetGpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations, float* error)
	{
		const auto maxIterations = configuration.MaxIterations.has_value() ? configuration.MaxIterations.value() : -1;
		const auto maxNicpRepetitions = maxIterations > 0 ? maxIterations : nicpIterDefault;
		const auto nicpSubcloudSize = before.size() < nicpSubcloudDefault ? before.size() : nicpSubcloudDefault;

		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				return GetCudaIcpTransformationMatrix(before, after, eps, maxIterations, iterations, error);
			case ComputationMethod::NoniterativeIcp:
				return GetCudaNicpTransformationMatrix(before, after, eps, maxNicpRepetitions, nicpBatchSize, configuration.ApproximationType, nicpSubcloudSize, iterations, error);
			case ComputationMethod::Cpd:
				return GetCudaCpdTransformationMatrix(before, after, eps, configuration.CpdWeight, cpdConstScale, maxIterations, eps, configuration.ApproximationType, iterations, error);
			default:
				assert(false); //unknown method
				return GetCudaIcpTransformationMatrix(before, after, eps, maxIterations, iterations, error);
		}
	}

	int RunGpuTests()
	{ 
		const auto methods = { ComputationMethod::Cpd };
		Tests::RunTestSet(GetSizesTestSet, GetGpuSlamResult, "sizes", methods);
		return 0;
	}
}

int main(int argc, char** argv)
{
#ifdef TEST
	return RunGpuTests();
#else
	return Main(argc, argv, GetGpuSlamResult);
#endif
}
