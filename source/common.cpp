#include <stdio.h>
#include "common.h"

namespace Common 
{
	void LibraryTest()
	{
		srand(666);
		Point_f corner = { -1, -1, -1 };
		Point_f size = { 2, 2, 2 };

		auto cloud = GetRandomPointCloud(corner, size, 1000);
		printf("Cloud generated\n");
	}

	std::vector<Point_f> GetRandomPointCloud(Point_f corner, Point_f size, int count)
	{
		const auto getRandomPoint = [&]() {
			Point_f point = {
				static_cast<float>(rand()) / RAND_MAX * size.x, 
				static_cast<float>(rand()) / RAND_MAX * size.y, 
				static_cast<float>(rand()) / RAND_MAX * size.z 
			};

			return point + corner;
		};

		std::vector<Point_f> result;
		for (int i = 0; i < count; i++)
			result.push_back(getRandomPoint());

		return result;
	}
}
