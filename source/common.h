#pragma warning(disable : 4996)
#include <Eigen/Dense>
#include "_common.h"

#include "renderer.h"
#include "shadertype.h"

namespace Common
{
	typedef std::tuple<std::vector<Point_f>, std::vector<Point_f>, std::vector<int>, std::vector<int>> CorrespondingPointsTuple;

	/// Loads point cloud from .obj file
	/// \param[in] path Relative path to file
	std::vector<Point_f> LoadCloud(const std::string& path);

	/// Returns random subcloud of given size
	std::vector<Point_f> GetSubcloud(const std::vector<Point_f>& cloud, int subcloudSize);

	/// Normalizes the input cloud so it fits in cube with side of given size 
	std::vector<Point_f> NormalizeCloud(const std::vector<Point_f>& cloud, float size);

	// Transform cloud helpers
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix);
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& scale);

	// Transform point helpers
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	Point_f TransformPoint(const Point_f& point, const glm::mat4& transformationMatrix);
	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& scale);

	// Mean squared error helpers
	// Given clouds should be in corresponding orders and cloudAfter cannot have bigger size than cloudBefore
	[[deprecated("Replaced by version with rotation matrix and translation vector")]]
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& correspondingIndexesBefore, const std::vector<int> correspondingIndexesAfter);
	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);

	/// Gets ceneter of mass of the given cloud, useful when aligning cloud
	Point_f GetCenterOfMass(const std::vector<Point_f>& cloud);
	
	/// Aligns the cloud to given center of mass
	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud, const Point_f& center_of_mass);

	// Eigen-Point_f compatibility helpers
	Eigen::Matrix3Xf GetMatrix3XFromPointsVector(const std::vector<Point_f>& points);
	Eigen::VectorXf GetVectorXFromPointsVector(const std::vector<float>& vector);
	Eigen::MatrixXf GetMatrixXFromPointsVector(const std::vector<float>& points, const int& rows, const int& cols);
	Eigen::Vector3f ConvertToEigenVector(const Point_f& point);

	// Eigen-glm compatibility helpers
	glm::mat3 ConvertRotationMatrix(const Eigen::Matrix3f& rotationMatrix);
	glm::vec3 ConvertTranslationVector(const Eigen::Vector3f& translationVector);

	/// Converts rotation and translation to single transformation matrix 
	glm::mat4 ConvertToTransformationMatrix(const glm::mat3& rotationMatrix, const glm::vec3& translationVector);

	// Printing different types of matrices 
	void PrintMatrix(Eigen::Matrix3f matrix);
	void PrintMatrix(const glm::mat4& matrix);
	void PrintMatrix(const glm::mat3& matrix);
	void PrintMatrix(const glm::mat3& matrix, const glm::vec3& vector);

	/// Gets corresponding points between two clouds
	/// \param maxDistanceSquares All points which have no correspondence closer than this parameter will be rejected
	/// \param parallel Determines parallel or sequential execution
	CorrespondingPointsTuple GetCorrespondingPoints(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter, float maxDistanceSquared, bool parallel);

	/// Performs SVD optimization between cloudBefore and cloudAfter. Note that the clouds should be in corresponding order
	/// \returns Rotation and translation pair
	std::pair<glm::mat3, glm::vec3> LeastSquaresSVD(const std::vector<Common::Point_f>& cloudBefore, const std::vector<Common::Point_f>& cloudAfter);

	/// Creates random permutation vector with values in range [0, size) 
	std::vector<int> GetRandomPermutationVector(int size);

	/// Creates permutation inverse to input parameter
	std::vector<int> InversePermutation(const std::vector<int>& permutation);

	/// Permutes input cloud with given permutation 
	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation);
}
