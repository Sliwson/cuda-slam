#pragma once

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <vector>
#include "point.h"

namespace Common
{
	using Point_f = Point<float>;
	
	void LibraryTest();
	std::vector<Point_f> GetRandomPointCloud(Point_f corner, Point_f size, int count);
}
