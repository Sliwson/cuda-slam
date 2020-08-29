#include "testrunner.h"
#include "common.h"
#include "timer.h"

namespace Common
{
	TestRunner::TestRunner(SlamFunc func, std::string file) : computeFunction(func), outputFile(file)
	{
		if (!outputFile.empty())
			fileHandle = fopen(outputFile.c_str(), "w+");

		if (fileHandle != nullptr)
		{
			fprintf(fileHandle, "test-no;cloud-size;rotation;translation;time(ms);iterations;error\n");
		}
	}

	TestRunner::~TestRunner()
	{
		if (fileHandle != nullptr)
			fclose(fileHandle);
	}

	void TestRunner::RunAll()
	{
		currentTestIndex = 0;
		while (!tests.empty())
		{
			printf("==================================================================\n");
			printf("Running test %d\n", currentTestIndex);
			printf("==================================================================\n");

			auto test = tests.front();
			tests.pop();
			RunSingle(test);

			printf("==================================================================\n");
			printf("Test ended\n");
			printf("==================================================================\n\n");

			currentTestIndex++;
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

		if (fileHandle != nullptr)
		{
			fprintf(
				fileHandle, 
				"%d;%zd;%f;%f;%lld;%d;%f\n",
				currentTestIndex,
				before.size(),
				configuration.TransformationParameters.has_value() ? configuration.TransformationParameters.value().first : -1.f,
				configuration.TransformationParameters.has_value() ? configuration.TransformationParameters.value().second : -1.f,
				timer.GetStageTime("test"),
				0,
				error
			);
		}
	}
}
