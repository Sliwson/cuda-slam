#include "_common.h"

#include "renderer.h"
#include "shadertype.h"

namespace Common
{
	void LibraryTest();
	std::vector<Point_f> LoadCloud(const std::string& path);
	glm::mat4 SolveLeastSquaresSvd(const glm::mat3& matrix, const glm::vec3& centroidBefore, const glm::vec3& centroidAfter);
	void PrintMatrix(const glm::mat4& matrix);
	void PrintMatrix(const glm::mat3& matrix);
}
