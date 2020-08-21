#include <stdio.h>
#include "tests.h"
#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"
#include "common.h"

int main(int argc, char** argv)
{
	auto configParser = Common::ConfigParser(argc, argv);
	if (!configParser.IsCorrect())
	{
		printf("Aborting\n");
		return -1;
	}

	auto configuration = configParser.GetConfiguration();
	configuration.Print();

	auto [before, after] = Common::GetCloudsFromConfig(configuration);

	//calculate
	auto result = [&, beforeCloud = before, afterCloud = after]() {
		switch (configuration.ComputationMethod) {
			case Common::ComputationMethod::Icp:
				return BasicICP::CalculateICPWithConfiguration(beforeCloud, afterCloud, configuration);
			case Common::ComputationMethod::Cpd:
				return NonIterative::CalculateNonIterativeWithConfiguration(beforeCloud, afterCloud, configuration);
			case Common::ComputationMethod::NoniterativeIcp:
				return BasicICP::CalculateICPWithConfiguration(beforeCloud, afterCloud, configuration);
		}
	}();

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
