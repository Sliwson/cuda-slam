#include <stdio.h>
#include "tests.h"

constexpr float TEST_EPS = 1e-6f;
constexpr int CLOUD_SIZE = 3000;
const char* object_path = "data/bunny.obj";

int main()
{
	printf("Hello cpu-slam!\n");
	Tests::NonIterativeTest(object_path, CLOUD_SIZE, TEST_EPS);
	//Tests::BasicICPTest(object_path, CLOUD_SIZE, TEST_EPS);
	return 0;
}
