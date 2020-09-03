#include "cpdcuda.cuh"
#include "nicpcuda.cuh"
#include "icpcuda.cuh"

#include "mainwrapper.h"

using namespace Common;

namespace {
	constexpr auto eps = 1e-5f;

	std::pair<glm::mat3, glm::vec3> GetGpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations)
	{
		const int maxIterations = configuration.MaxIterations.has_value() ? configuration.MaxIterations.value() : -1;

		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				return GetCudaIcpTransformationMatrix(before, after, iterations, eps, maxIterations);
			case ComputationMethod::NoniterativeIcp:
			case ComputationMethod::Cpd:
			default:
				assert(false); //unknown method
		}

		return { glm::mat3(0), glm::vec3(0) };
	}

	int RunGpuTests()
	{ 
		const auto methods = { ComputationMethod::Icp };
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
