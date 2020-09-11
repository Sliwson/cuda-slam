#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"

#include "mainwrapper.h"
#include "common.h"

using namespace Common;

namespace {
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations, float* error)
	{
		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations, error);
			case ComputationMethod::NoniterativeIcp:
				return NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration, iterations, error);
			case ComputationMethod::Cpd:
				return CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration, iterations, error);
			default:
				assert(false); //unknown method
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations, error);
		}
	}

	int RunCpuTests()
	{ 
		srand(Tests::RANDOM_SEED);
		Common::SetRandom();

		const auto methods = { ComputationMethod::Icp, ComputationMethod::NoniterativeIcp, ComputationMethod::Cpd };
		Tests::RunTestSet(GetSizesTestSet, GetCpuSlamResult, "sizes", methods);
		return 0;
	}
}

int main(int argc, char** argv)
{
#ifdef TEST
	return RunCpuTests();
#else
	return Main(argc, argv, "Cpu Slam", GetCpuSlamResult);
#endif
}
