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
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const Common::CpuCloud& before, const Common::CpuCloud& after, Configuration configuration, int* iterations)
	{
		switch (configuration.ComputationMethod) {
			case Common::ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations);
			case Common::ComputationMethod::Cpd:
				return NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration, iterations);
			case Common::ComputationMethod::NoniterativeIcp:
				return CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration, iterations);
			default:
				assert(false); //unknown method
		}
	}

	int RunCpuTests()
	{ 
		const auto run_test_set = [](std::function<std::vector<Configuration>()> acquireFunc, std::string name) {
			auto testSet = acquireFunc();
			auto runner = TestRunner(GetCpuSlamResult, name);
		
			for (const auto& test : testSet)
			runner.AddTest(test);

			runner.RunAll();
		};

		run_test_set(Common::GetBasicTestSet, "basic.csv");
		run_test_set(Common::GetSizesTestSet, "sizes.csv");

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
		int iterations = 0;
		auto result = GetCpuSlamResult(before, after, configuration, &iterations);

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
