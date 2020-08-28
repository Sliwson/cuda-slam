#include "testrunner.h"

void Common::TestRunner::RunAll()
{
	while (!tests.empty())
	{
		auto test = tests.front();
		tests.pop();
		RunSingle(test);
	}
}

void Common::TestRunner::RunSingle(Configuration config)
{
}
