#include <stdio.h>
#include "common.h"

int main()
{
	printf("Hello cpu-slam!\n");
	//Common::LibraryTest();

	const int size = 10;

	const auto cloud = Common::LoadCloud("data/bunny.obj");


	std::vector<Common::Point_f> origin_points(size);
	std::vector<Common::Point_f> result_points(size);
	std::vector<Common::Point_f> cpu_points(size);
	std::vector<Common::Point_f> gpu_points(size);
	std::vector<Common::Point_f> empty_points(1);

	//empty_points.push_back(Common::Point_f::Zero());

	//empty_points.push_back

	for (int i = 0; i < size; i++)
	{
		origin_points[i] = Common::Point_f(i, 0, 0);
		result_points[i] = Common::Point_f(0, i, 0);
		cpu_points[i] = Common::Point_f(i, -i, i);
		gpu_points[i] = Common::Point_f(0, 0, i);
	}	

	Common::Renderer renderer(
		Common::ShaderType::simpleModel,
		cloud,
		cloud,
		cloud,
		cloud);

	renderer.Show();

	return 0;
}
