#include "mainwrapper.h"

namespace Common
{
	int Main(int argc, char** argv, const char* windowName, const SlamFunc& func)
	{
		auto configParser = ConfigParser(argc, argv);
		if (!configParser.IsCorrect())
		{
			printf("Aborting\n");
			return -1;
		}

		Configuration configuration = configParser.GetConfiguration();
		configuration.Print();

		if (configuration.RandomSeed.has_value())
		{
			srand(configuration.RandomSeed.value());
		}
		else
		{
			srand(time(nullptr));
		}

		auto [before, after] = GetCloudsFromConfig(configuration);

		//calculate
		int iterations = 0;
		float error = 0.f;
		auto result = std::make_pair(glm::mat3(1.0f), glm::vec3(0.f, 10.0f, 0.0f));//func(before, after, configuration, &iterations, &error);

		const auto vec = result.second;

		printf("Results:\n");
		printf("Rotation matrix:\n");
		PrintMatrix(result.first);
		printf("Translation vector:\n");
		printf("x = %f, y = %f, z = %f\n", vec.x, vec.y, vec.z);
		printf("Error: %f\n", error);
		printf("Before Size: %d After size %d\n", before.size(), after.size());

		auto resultCloud = GetTransformedCloud(before, result.first, result.second);

		// visualisation
		if (configuration.ShowVisualisation)
		{
			auto renderer = Renderer(
				ShaderType::SimpleModel,
				windowName,
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
