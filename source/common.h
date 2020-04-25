#include "_common.h"

namespace Common
{
	std::vector<Point_f> LoadCloud(const std::string& path);
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix);
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& correspondingIndexesBefore, const std::vector<int> correspondingIndexesAfter);
	Point_f GetCenterOfMass(const std::vector<Point_f>& cloud);
	Eigen::Matrix3Xf GetMatrix3XFromPointsVector(const std::vector<Point_f>& points);
	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud, const Point_f& center_of_mass);
	glm::mat3 ConvertRotationMatrix(const Eigen::Matrix3f& rotationMatrix);
	glm::vec3 ConvertTranslationVector(const Eigen::Vector3f& translationVector);
	Point_f TransformPoint(const Point_f& point, const glm::mat4& transformationMatrix);
	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
}
