#include "common.h"

#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"

#include "configparser.h"
#include "configuration.h"
#include "testrunner.h"
#include "testset.h"
#include "testutils.h"

using namespace Common;

namespace {
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const ::CpuCloud& before, const ::CpuCloud& after, Configuration configuration, int* iterations)
	{
		switch (configuration.ComputationMethod) {
			case ::ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations);
			case ::ComputationMethod::NoniterativeIcp:
				return NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration, iterations);
			case ::ComputationMethod::Cpd:
				return CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration, iterations);
			default:
				assert(false); //unknown method
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations);
		}
	}

	int RunCpuTests()
	{ 
		const auto methods = { ComputationMethod::Icp, ComputationMethod::NoniterativeIcp, ComputationMethod::Cpd };
		Tests::RunTestSet(::GetSizesTestSet, GetCpuSlamResult, "sizes", methods);
		return 0;
	}

	int CpuMain(int argc, char** argv)
	{
		auto configParser = ::ConfigParser(argc, argv);
		if (!configParser.IsCorrect())
		{
			printf("Aborting\n");
			return -1;
		}

		::Configuration configuration = configParser.GetConfiguration();
		configuration.Print();

		auto [before, after] = ::GetCloudsFromConfig(configuration);

		//calculate
		int iterations = 0;
		auto result = GetCpuSlamResult(before, after, configuration, &iterations);

		auto resultCloud = ::GetTransformedCloud(before, result.first, result.second);

		// visualisation
		if (configuration.ShowVisualisation)
		{
			auto renderer = Renderer(
				::ShaderType::SimpleModel,
				before,
				after,
				resultCloud,
				{ Point_f::Zero() }
			);

			renderer.Show();
		}

		return 0;
	}
}

int main(int argc, char** argv)
{
#ifdef TEST
	return RunCpuTests();
#else
	return CpuMain(argc, argv);
#endif
}
