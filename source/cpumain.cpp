#include "common.h"

#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"

#include "configparser.h"
#include "configuration.h"
#include "testrunner.h"
#include "testset.h"

#include "tests.h"

//#define TEST

namespace {
	std::pair<glm::mat3, glm::vec3> GetCpuSlamResult(const Common::CpuCloud& before, const Common::CpuCloud& after, Configuration configuration, int* iterations)
	{
		switch (configuration.ComputationMethod) {
			case Common::ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(before, after, configuration, iterations);
			case Common::ComputationMethod::NoniterativeIcp:
				return NonIterative::CalculateNonIterativeWithConfiguration(before, after, configuration, iterations);
			case Common::ComputationMethod::Cpd:
				return CoherentPointDrift::CalculateCpdWithConfiguration(before, after, configuration, iterations);
			default:
				assert(false); //unknown method
		}
	}

	int RunCpuTests()
	{ 
		static_assert(static_cast<int>(ComputationMethod::Icp) == 0);
		static_assert(static_cast<int>(ComputationMethod::NoniterativeIcp) == 1);
		static_assert(static_cast<int>(ComputationMethod::Cpd) == 2);

		const auto run_test_set = [](std::function<std::vector<Configuration>(ComputationMethod)> acquireFunc, std::string name) {
			const std::vector<std::string> methods = { "icp", "nicp", "cpd" };

			for (int i = 0; i < methods.size(); i++)
			{
				auto testSet = acquireFunc(ComputationMethod(i));
				const auto fileName = name + "-" + methods[i] + ".csv";
				auto runner = TestRunner(GetCpuSlamResult, fileName);

				for (const auto& test : testSet)
					runner.AddTest(test);

				runner.RunAll();
			}
		};

		run_test_set(Common::GetSizesTestSet, "sizes");

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
	//return CpuMain(argc, argv);
#endif

	const char* objectPath = "data/bunny.obj";
	int pointCount = -1;
	float testEps = 1e-4f;
	float weight = 0.1f;
	bool const_scale = false;
	const int max_iterations = 50;
	Common::ApproximationType fgt = Common::ApproximationType::Full;

	Tests::RigidCPDTest(objectPath, pointCount, testEps, weight, const_scale, max_iterations, fgt);
}
