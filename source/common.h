#pragma warning(disable : 4996)
#include <Eigen/Dense>
#include "_common.h"

#include "renderer.h"
#include "shadertype.h"

namespace Common
{
	typedef std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>> CorrespondingPointsTuple;

	std::vector<Point_f> LoadCloud(const std::string& path);
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix);
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& correspondingIndexesBefore, const std::vector<int> correspondingIndexesAfter);
	Point_f GetCenterOfMass(const std::vector<Point_f>& cloud);
	Eigen::Matrix3Xf GetMatrix3XFromPointsVector(const std::vector<Point_f>& points);
	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud, const Point_f& center_of_mass);
	glm::mat3 ConvertRotationMatrix(const Eigen::Matrix3f& rotationMatrix);
	glm::vec3 ConvertTranslationVector(const Eigen::Vector3f& translationVector);
	glm::mat4 ConvertToTransformationMatrix(const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	Point_f TransformPoint(const Point_f& point, const glm::mat4& transformationMatrix);
	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	void PrintMatrix(Eigen::Matrix3f matrix);
	void PrintMatrix(const glm::mat4& matrix);
	void PrintMatrix(const glm::mat3& matrix);
	void PrintMatrix(const glm::mat3& matrix, const glm::vec3& vector);

	CorrespondingPointsTuple GetCorrespondingPoints(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, float maxDistanceSquared);
	std::pair<glm::mat3, glm::vec3> LeastSquaresSVD(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter);
}
