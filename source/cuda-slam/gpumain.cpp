#include "cpdcuda.cuh"
#include "nicpcuda.cuh"
#include "icpcuda.cuh"

#include "mainwrapper.h"

using namespace Common;

#define TEST

namespace {
	constexpr auto eps = 1e-5f;

	//todo move nicp config to configuration class
	constexpr auto nicpIterDefault = 20;
	constexpr auto nicpSubcloudDefault = 1000;
	constexpr auto nicpBatchSize = 16;

	std::pair<glm::mat3, glm::vec3> GetGpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations)
	{
		const auto maxIterations = configuration.MaxIterations.has_value() ? configuration.MaxIterations.value() : -1;
		const auto maxNicpRepetitions = maxIterations > 0 ? maxIterations : nicpIterDefault;
		const auto nicpSubcloudSize = before.size() < nicpSubcloudDefault ? before.size() : nicpSubcloudDefault;

		float error = 0.f;
		
		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				return GetCudaIcpTransformationMatrix(before, after, iterations, eps, maxIterations);
			case ComputationMethod::NoniterativeIcp:
				return GetCudaNicpTransformationMatrix(before, after, iterations, &error, eps, maxNicpRepetitions, nicpBatchSize, configuration.ApproximationType, nicpSubcloudSize);
			case ComputationMethod::Cpd:
			default:
				assert(false); //unknown method
		}

		return { glm::mat3(0), glm::vec3(0) };
	}

	int RunGpuTests()
	{ 
		const auto methods = { ComputationMethod::NoniterativeIcp };
		Tests::RunTestSet(::GetSizesTestSet, GetGpuSlamResult, "sizes", methods);
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
