#include <stdio.h>
#include "tests.h"
#include "coherentpointdrift.h"
#include "noniterative.h"
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

	// visualisation
	if (configuration.ShowVisualisation)
	{
		auto renderer = Renderer(
			Common::ShaderType::SimpleModel,
			before,
			after,
			after,
			{ Point_f::Zero() }
		);

		renderer.Show();
	}

	return 0;
}
