#include <stdio.h>
#include "tests.h"
#include "coherentpointdrift.h"

constexpr float TEST_EPS = 1e-6f;
constexpr int CLOUD_SIZE = 3000;
const char* object_path1 = "data/bunny.obj";
const char* object_path2 = "data/bunny-decapitated.obj";
const char* object_path3 = "data/bunny-head.obj";
const char* object_path4 = "data/bunny-faceless.obj";
const char* object_path5 = "data/bunny-tailless.obj";
const char* object_path6 = "data/noise_00_bunny.off";
const char* object_path7 = "data/noise_25_bunny.off";
const char* object_path8 = "data/noise_50_bunny.off";

int main()
{
	printf("Hello cpu-slam!\n");
	const float outliers = 0.5f;
	const bool const_scale = false;
	const int max_iter = 50;
	const int fgt = 1; //0-not use fgt, 1-use fgt, 2-hybrid
	Tests::RigidCPDTest(object_path4, object_path5, -1, -1, TEST_EPS, outliers, const_scale, max_iter, fgt);
	//Tests::BasicICPTest(object_path4, object_path5, -1, -1, TEST_EPS);

	return 0;
}
