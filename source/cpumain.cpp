#include <stdio.h>
#include "tests.h"
#include "fgt.h"
#include "fgt_model.h"

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

struct FGT_model;

int main()
{
	printf("Hello cpu-slam!\n");
	//Tests::BasicICPTest(object_path2, object_path1, -1, -1, TEST_EPS);

	auto cloud = std::vector<Point_f>();

	cloud.push_back(Point_f(0.309336386192094,
		0.707084625951691,
		0.298113288774646));
	cloud.push_back(Point_f(0.927420719822510,
		1.13723783523452,
		- 1.06097318210616));
	cloud.push_back(Point_f(-0.193269258310719,
		- 1.04186474555901,
		- 1.17995754544710));
	cloud.push_back(Point_f(0.0171965470714722,
		- 1.24309161063860,
		1.08001918998961));
	cloud.push_back(Point_f(-0.835916847502900,
		0.492095628437480,
		- 0.372592706818986));
	cloud.push_back(Point_f(1.09689261405179,
		- 1.96878196118039,
		- 1.81027465424799));
	cloud.push_back(Point_f(-1.61339867481782,
		- 1.44189376201321,
		- 1.12133640172003));
	cloud.push_back(Point_f(1.05062755717162,
		- 0.409510045836082,
		- 1.59301882701997));
	cloud.push_back(Point_f(0.968175497182022,
		1.51454386177120,
		0.181131970076453));
	cloud.push_back(Point_f(-0.331322753540273,
		0.192331269482018,
		- 0.516441645483671));

	auto weights = std::vector(10, 0.0f);

	weights[1] = 1.0f;

	float sigma = 0.5f;
	int k = 5;
	int p = 2;

	auto res = FastGaussTransform::ComputeFGTModel(cloud, weights, sigma, k, p);

	std::cout << "XCvector:" << std::endl;
	for (const auto& point : res.xc)
	{
		printf("[ %g , %g , %g ]\n", point.x, point.y, point.z);
	}
	std::cout << "A_K matrix:" << std::endl;
	std::cout << res.Ak << std::endl;

	return 0;
}
