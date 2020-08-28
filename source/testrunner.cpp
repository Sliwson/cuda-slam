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
		const auto [before, after] = Common::GetCloudsFromConfig(configuration);

		auto timer = Common::Timer();

		timer.StartStage("test");
		const auto result = computeFunction(before, after, configuration);
		timer.StopStage("test");
		timer.PrintResults();

		const auto resultCloud = Common::GetTransformedCloud(before, result.first, result.second);
	}
}
