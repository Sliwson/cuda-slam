#include "cpdcuda.cuh"
#include "nicpcuda.cuh"
#include "icpcuda.cuh"

#include "mainwrapper.h"

using namespace Common;

namespace {
	std::pair<glm::mat3, glm::vec3> GetGpuSlamResult(const ::CpuCloud& before, const ::CpuCloud& after, Configuration configuration, int* iterations)
	{
		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
			case ComputationMethod::NoniterativeIcp:
			case ComputationMethod::Cpd:
			default:
				assert(false); //unknown method
		}
	}

	int RunGpuTests()
	{ 
		const auto methods = { ComputationMethod::Icp, ComputationMethod::NoniterativeIcp, ComputationMethod::Cpd };
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
