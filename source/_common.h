#pragma once

#pragma comment (lib, "glfw3")

#ifdef _DEBUG
	#pragma comment (lib, "assimp-vc142-mtd")
#else
	#pragma comment (lib, "assimp-vc142-mt")
#endif

#include <vector>
#include <filesystem>
#include <algorithm>
#include <stdio.h>
#include <numeric>
#include <random>
#include <iostream>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "point.h"

namespace Common
{
	using Point_f = Point<float>;
}
