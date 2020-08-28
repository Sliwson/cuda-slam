#include "testrunner.h"
#include "common.h"
#include "timer.h"

namespace Common
{
	void TestRunner::RunAll()
	{
		int testIdx = 0;
		while (!tests.empty())
		{
			printf("==================================================================\n");
			printf("Running test %d\n", testIdx);
			printf("==================================================================\n");

			auto test = tests.front();
			tests.pop();
			RunSingle(test);

			printf("==================================================================\n");
			printf("Test ended\n");
			printf("==================================================================\n\n");

			testIdx++;
		}
	}

	void TestRunner::RunSingle(Configuration configuration)
	{
		const auto [before, after] = GetCloudsFromConfig(configuration);

		auto timer = Common::Timer();

		timer.StartStage("test");
		const auto result = computeFunction(before, after, configuration);
		timer.StopStage("test");
		timer.PrintResults();

		const auto resultCloud = GetTransformedCloud(before, result.first, result.second);
		const auto correspondingPoints = GetCorrespondingPoints(resultCloud, after, configuration.MaxDistanceSquared, true);
		const auto error = GetMeanSquaredError(resultCloud, after, std::get<2>(correspondingPoints), std::get<3>(correspondingPoints));

		printf("Error: %f\n", error);
	}
}
