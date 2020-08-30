#include <stdio.h>
#include "tests.h"
#include "coherentpointdrift.h"
#include "noniterative.h"
#include "basicicp.h"
#include "common.h"
#include "configparser.h"
#include "configuration.h"

int main(int argc, char** argv)
{
	//auto configParser = Common::ConfigParser(argc, argv);
	//if (!configParser.IsCorrect())
	//{
	//	printf("Aborting\n");
	//	return -1;
	//}

	//Common::Configuration configuration = configParser.GetConfiguration();
	//configuration.Print();

	//auto [before, after] = Common::GetCloudsFromConfig(configuration);

	////calculate
	//auto result = [&, beforeCloud = before, afterCloud = after]() {
	//	switch (configuration.ComputationMethod) {
	//		case Common::ComputationMethod::Icp:
	//			return BasicICP::CalculateICPWithConfiguration(beforeCloud, afterCloud, configuration);
	//		case Common::ComputationMethod::Cpd:
	//			return NonIterative::CalculateNonIterativeWithConfiguration(beforeCloud, afterCloud, configuration);
	//		case Common::ComputationMethod::NoniterativeIcp:
	//			return CoherentPointDrift::CalculateCpdWithConfiguration(beforeCloud, afterCloud, configuration);
	//		default:
	//			assert(false, "Unknown method");
	//	}
	//}();

	//auto resultCloud = Common::GetTransformedCloud(before, result.first, result.second);

	//// visualisation
	//if (configuration.ShowVisualisation)
	//{
	//	auto renderer = Renderer(
	//		Common::ShaderType::SimpleModel,
	//		before,
	//		after,
	//		resultCloud,
	//		{ Point_f::Zero() }
	//	);

	//	renderer.Show();
	//}

	//michal things
	const char* objectPathBunny = "data/bunny.obj";
	const int pointCount = -1;
	const float testEps = 1e-4f;
	const float weight = 0.1f;
	const bool const_scale = false;
	const int max_iterations = 50;
	const Common::ApproximationType fgt = Common::ApproximationType::None;

	//Tests::RigidCPDTest(cloud, TEST_EPS, outliers, const_scale, max_iter, fgt);
	Tests::RigidCPDTest(objectPathBunny, pointCount, testEps, weight, const_scale, max_iterations, fgt);

	return 0;
}
