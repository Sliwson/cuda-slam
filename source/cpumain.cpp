#include <stdio.h>
#include "tests.h"

constexpr float TEST_EPS = 1e-6f;
constexpr int CLOUD_SIZE = 3000;
const char* object_path = "data/bunny.obj";
const char* object_path1 = "data/noise_00_bunny.off";
const char* object_path2 = "data/noise_25_bunny.off";
const char* object_path3 = "data/noise_50_bunny.off";
const char* object_path4 = "data/bunny-decapitated.obj";

int main()
{
	printf("Hello cpu-slam!\n");
	Tests::BasicRigidCPDTest(object_path, CLOUD_SIZE, TEST_EPS);
	return 0;
}
