#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"

#include "mainwrapper.h"
#include "common.h"

using namespace Common;

#define TEST

namespace {
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const CpuCloud& before, const CpuCloud& after, Configuration configuration, int* iterations, float* error)
	{
		std::pair<glm::mat3, glm::vec3> result;
		switch (configuration.ComputationMethod) {
			case ComputationMethod::Icp:
				result = BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations, error);
				break;
			case ComputationMethod::NoniterativeIcp:
				result = NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration, iterations, error);
				break;
			case ComputationMethod::Cpd:
				result = CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration, iterations, error);
				break;
			default:
				assert(false); //unknown method
				printf("Invalid method\n");
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations, error);
		}

		const auto transformedCloud = GetTransformedCloud(before, result.first, result.second);
		const auto correspondences = GetCorrespondingPoints(transformedCloud, after, configuration.MaxDistanceSquared, true);
		*error = GetMeanSquaredError(std::get<0>(correspondences), std::get<1>(correspondences));

		return result;
	}

	int RunCpuTests()
	{ 
		srand(Tests::RANDOM_SEED);
		Common::SetRandom();

		const auto methods = { ComputationMethod::Cpd };
		Tests::RunTestSet(GetConvergenceTestSet, GetCpuSlamResult, "convergence", methods);
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
