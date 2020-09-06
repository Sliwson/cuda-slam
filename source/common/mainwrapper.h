#pragma once

#include "common.h"
#include "configparser.h"
#include "configuration.h"
#include "testrunner.h"
#include "testset.h"
#include "testutils.h"

namespace Common
{
	int Main(int argc, char** argv, const char* windowName, const SlamFunc& func);
}
