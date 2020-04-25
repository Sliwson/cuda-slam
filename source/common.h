#include "_common.h"

#include "renderer.h"
#include "shadertype.h"

namespace Common
{
	void LibraryTest();
	std::vector<Point_f> LoadCloud(const std::string& path);
	glm::mat4 GetTransform(glm::mat3 forSvd, glm::vec3 b, glm::vec3 a);
	void PrintMatrix(const glm::mat4& matrix);
	void PrintMatrix(const glm::mat3& matrix);
}
