#pragma once

#define WIN32_LEAN_AND_MEAN

#ifdef _DEBUG
	#pragma comment (lib, "assimp-vc142-mtd")
	#pragma comment (lib, "glfw3")

#else
	#pragma comment (lib, "assimp-vc142-mt")
	#pragma comment (lib, "glfw3")
#endif


#include <vector>
#include "point.h"

namespace Common
{
	using Point_f = Point<float>;

	void LibraryTest();
}
