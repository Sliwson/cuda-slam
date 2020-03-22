#include <stdio.h>
#include "common.h"

int main()
{
	printf("Hello cpu-slam!\n");
	//Common::LibraryTest();

	const int size = 10;

	std::vector<Common::Point_f> origin_points(size);
	std::vector<Common::Point_f> result_points(size);
	std::vector<Common::Point_f> cpu_points(size);
	std::vector<Common::Point_f> gpu_points(size);

	for (int i = 0; i < size; i++)
	{
		origin_points[i] = Common::Point_f(i, 0, 0);
		result_points[i] = Common::Point_f(0, 100*i, 0);
		cpu_points[i] = Common::Point_f(i, -i, i);
		gpu_points[i] = Common::Point_f(0, 0, i);
	}

	

	Common::Renderer renderer(
		Common::ShaderType::simpleModel,
		origin_points,
		result_points,
		cpu_points,
		gpu_points);

	renderer.Show();

	return 0;
}
