#include "testrunner.h"
#include "common.h"

namespace Common
{
	void TestRunner::RunAll()
	{
		while (!tests.empty())
		{
			auto test = tests.front();
			tests.pop();
			RunSingle(test);
		}
	}

	void TestRunner::RunSingle(Configuration configuration)
	{
		auto [before, after] = Common::GetCloudsFromConfig(configuration);
		auto result = computeFunction(before, after, configuration);
		auto resultCloud = Common::GetTransformedCloud(before, result.first, result.second);
	}
}
