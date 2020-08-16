#pragma once

#pragma comment (lib, "glfw3")

#ifdef _DEBUG
	#pragma comment (lib, "assimp-vc142-mtd")
#else
	#pragma comment (lib, "assimp-vc142-mt")
#endif

#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <stdio.h>
#include <numeric>
#include <random>
#include <iostream>
#include <functional>
#include <optional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "point.h"

namespace Common
{
	constexpr int DIMENSION = 3;
	using Point_f = Point<float>;

	void PrintMatrix(const glm::mat3& matrix);
}
