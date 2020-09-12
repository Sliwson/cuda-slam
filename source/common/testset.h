#pragma once
#include "common.h"

namespace Common
{
	std::vector<Configuration> GetSizesTestSet(ComputationMethod method);
	std::vector<Configuration> GetPerformanceTestSet(ComputationMethod method);
	std::vector<Configuration> GetConvergenceTestSet(ComputationMethod method);
}
