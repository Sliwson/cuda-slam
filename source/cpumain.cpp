#include <stdio.h>
#include "tests.h"
#include "coherentpointdrift.h"
#include "noniterative.h"
#include "common.h"

constexpr float TEST_EPS = 1e-6f;
constexpr int CLOUD_SIZE = 10000;
const char* objectPathBunny = "data/bunny.obj";
const char* objectPathBunnyDecapitated = "data/bunny-decapitated.obj";
const char* objectPathBunnyHead = "data/bunny-head.obj";
const char* objectPathBunnyFaceless = "data/bunny-faceless.obj";
const char* objectPathBunnyTailless = "data/bunny-tailless.obj";
const char* objectPathBunnyNoise00 = "data/noise_00_bunny.off";
const char* objectPathBunnyNoise25 = "data/noise_25_bunny.off";
const char* objectPathBunnyNoise50 = "data/noise_50_bunny.off";
const char* objectPathRose = "data/rose.obj";

int main(int argc, char** argv)
{
	auto configParser = Common::ConfigParser(argc, argv);
	if (!configParser.IsCorrect())
	{
		printf("Aborting\n");
		return -1;
	}

	printf("Hello cpu-slam!\n");
	auto configuration = configParser.GetConfiguration();
	configuration.Print();
	return 0;

	//const float outliers = 0.5f;
	//const bool const_scale = false;
	//const int max_iter = 50;
	//const auto fgt = FastGaussTransform::FGTType::Full; //None-not use fgt, Full-use fgt, Hybrid-hybrid
	//Tests::RigidCPDTest(objectPathBunnyFaceless, objectPathBunnyTailless, -1, -1, TEST_EPS, outliers, const_scale, max_iter, fgt);
	//Tests::BasicICPTest(object_path4, object_path5, -1, -1, TEST_EPS);

	const auto type = NonIterative::NonIterativeApproximation::None;
//	Tests::NonIterativeTest(objectPathBunny, -1, TEST_EPS, 20, type, CLOUD_SIZE);
	Tests::BasicICPTest(objectPathBunny, CLOUD_SIZE, TEST_EPS);
	return 0;
}
