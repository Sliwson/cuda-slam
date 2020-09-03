#include "mainwrapper.h"

namespace Common
{
	int Main(int argc, char** argv, const SlamFunc& func)
	{
		auto configParser = ConfigParser(argc, argv);
		if (!configParser.IsCorrect())
		{
			printf("Aborting\n");
			return -1;
		}

		Configuration configuration = configParser.GetConfiguration();
		configuration.Print();

		auto [before, after] = GetCloudsFromConfig(configuration);

		//calculate
		int iterations = 0;
		auto result = func(before, after, configuration, &iterations);

		auto resultCloud = GetTransformedCloud(before, result.first, result.second);

		// visualisation
		if (configuration.ShowVisualisation)
		{
			auto renderer = Renderer(
				ShaderType::SimpleModel,
				before,
				after,
				resultCloud,
				{ Point_f::Zero() }
			);

			renderer.Show();
		}

		return 0;
	}
}
