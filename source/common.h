#pragma once

#define WIN32_LEAN_AND_MEAN

#ifdef _DEBUG
	#pragma comment (lib, "assimp-vc142-mtd")
#else
	#pragma comment (lib, "assimp-vc142-mt")
#endif


#include <vector>
#include "point.h"

namespace Common
{
	using Point_f = Point<float>;

	void LibraryTest();
	void MassCenterTest();
}
