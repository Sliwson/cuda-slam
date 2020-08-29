#pragma once
#include "common.h"

namespace Common
{
	std::vector<Configuration> GetBasicTestSet();
	std::vector<Configuration> GetSizesTestSet(ComputationMethod method);
}
