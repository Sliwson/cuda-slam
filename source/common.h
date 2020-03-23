#pragma once

#define WIN32_LEAN_AND_MEAN

#pragma comment (lib, "glfw3")

#ifdef _DEBUG
	#pragma comment (lib, "assimp-vc142-mtd")
#else
	#pragma comment (lib, "assimp-vc142-mt")
#endif


#include <vector>
#include "point.h"
#include "renderer.h"
#include "shadertype.h"

namespace Common
{
	using Point_f = Point<float>;

	void LibraryTest();
	std::vector<Point_f> LoadCloud(const std::string& path);
}
