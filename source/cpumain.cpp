#include "common.h"

#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"

#include "configparser.h"
#include "configuration.h"
#include "testrunner.h"
#include "testset.h"

#define TEST

namespace {
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const Common::cpu_cloud& before, const Common::cpu_cloud& after, Configuration configuration)
	{
		switch (configuration.ComputationMethod) {
			case Common::ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration);
			case Common::ComputationMethod::Cpd:
				return NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration);
			case Common::ComputationMethod::NoniterativeIcp:
				return CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration);
			default:
				assert(false, "Unknown method");
		}
	}

	int RunCpuTests()
	{ 
		auto testSet = Common::GetBasicTestSet();
		auto runner = TestRunner(GetCpuSlamResult);

		for (const auto& test : testSet)
			runner.AddTest(test);

		runner.RunAll();

		return 0;
	}

	int CpuMain(int argc, char** argv)
	{
		auto configParser = Common::ConfigParser(argc, argv);
		if (!configParser.IsCorrect())
		{
			printf("Aborting\n");
			return -1;
		}

		Common::Configuration configuration = configParser.GetConfiguration();
		configuration.Print();

		auto [before, after] = Common::GetCloudsFromConfig(configuration);

		//calculate
		auto result = GetCpuSlamResult(before, after, configuration);

		auto resultCloud = Common::GetTransformedCloud(before, result.first, result.second);

		// visualisation
		if (configuration.ShowVisualisation)
		{
			auto renderer = Renderer(
				Common::ShaderType::SimpleModel,
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
