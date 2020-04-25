#include <chrono>
#include <tuple>

#include "common.h"
#include "loader.h"

namespace Common
{
	std::vector<Point_f> LoadCloud(const std::string& path)
	{
		AssimpCloudLoader loader(path);
		if (loader.GetCloudCount() > 0)
			return loader.GetCloud(0);
		else
			return std::vector<Point_f>();
	}

	//i'll leave it here but check later if you can use svd from basicicp
	glm::mat4 SolveLeastSquaresSvd(Eigen::Matrix3f matrix, const glm::vec3& centroidBefore, const glm::vec3& centroidAfter)
	{
		Eigen::JacobiSVD<Eigen::Matrix3f> const svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		double det = (svd.matrixU() * svd.matrixV().transpose()).determinant();
		Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, det);
		Eigen::Matrix3f rotation = svd.matrixU() * diag * svd.matrixV().transpose();

		glm::mat4 transformationMatrix(0);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				transformationMatrix[i][j] = rotation(j, i);
			}
			transformationMatrix[i][3] = 0;
		}

		Eigen::Vector3f beforep{ centroidBefore.x, centroidBefore.y, centroidBefore.z };
		Eigen::Vector3f afterp{ centroidAfter.x, centroidAfter.y, centroidAfter.z };
		Eigen::Vector3f translation = afterp - (rotation * beforep);

		transformationMatrix[3][0] = translation.x();
		transformationMatrix[3][1] = translation.y();
		transformationMatrix[3][2] = translation.z();
		transformationMatrix[3][3] = 1.0f;
		return transformationMatrix;
	}

	glm::mat4 SolveLeastSquaresSvd(const glm::mat3& forSvd, const glm::vec3& centroidBefore, const glm::vec3& centroidAfter)
	{
		Eigen::Matrix3f matrix;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				matrix(i, j) = forSvd[j][i];

		return SolveLeastSquaresSvd(matrix, centroidBefore, centroidAfter);
	}

	Point_f TransformPoint(const Point_f& point, const glm::mat4& transformationMatrix)
	{
		const glm::vec3 result = transformationMatrix * glm::vec4(glm::vec3(point), 1.0f);
		return Point_f(result);
	}

	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		const glm::vec3 result = (rotationMatrix * point) + translationVector;
		return Point_f(result);
	}

	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](Point_f p) { return TransformPoint(p, matrix); });
		return clone;
	}

	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](const Point_f& p) { return TransformPoint(p, rotationMatrix, translationVector); });
		return clone;
	}

	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		float diffSum = 0.0f;
		// We assume clouds are the same size but if error is significant, you might want to check it
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto transformed = TransformPoint(cloudBefore[i], matrix);
			const auto diff = cloudAfter[i] - transformed;
			diffSum += diff.LengthSquared();
		}
		return diffSum / cloudBefore.size();
	}

	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		float diffSum = 0.0f;
		// We assume clouds are the same size but if error is significant, you might want to check it
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto transformed = TransformPoint(cloudBefore[i], rotationMatrix, translationVector);
			const auto diff = cloudAfter[i] - transformed;
			diffSum += diff.LengthSquared();
		}
		return diffSum / cloudBefore.size();
	}

	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& correspondingIndexesBefore, const std::vector<int> correspondingIndexesAfter)
	{
		float diffSum = 0.0f;
		for (int i = 0; i < correspondingIndexesBefore.size(); i++)
		{
			const auto diff = cloudAfter[correspondingIndexesAfter[i]] - cloudBefore[correspondingIndexesBefore[i]];
			diffSum += diff.LengthSquared();
		}
		return diffSum / correspondingIndexesBefore.size();
	}

	Point_f GetCenterOfMass(const std::vector<Point_f>& cloud)
	{
		return std::accumulate(cloud.begin(), cloud.end(), Point_f::Zero()) / (float)cloud.size();
	}

	// Return matrix with every column storing one point (in 3 rows)
	Eigen::Matrix3Xf GetMatrix3XFromPointsVector(const std::vector<Point_f>& points)
	{
		Eigen::Matrix3Xf result = Eigen::ArrayXXf::Zero(3, points.size());
		for (int i = 0; i < points.size(); i++)
		{
			result(0, i) = points[i].x;
			result(1, i) = points[i].y;
			result(2, i) = points[i].z;
		}
		return result;
	}

	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud, const Point_f& center_of_mass)
	{
		auto result = std::vector<Point_f>(cloud.size());
		std::transform(cloud.begin(), cloud.end(), result.begin(),
			[&center_of_mass](const Point_f& point) -> Point_f { return point - center_of_mass; });
		return result;
	}

	glm::mat3 ConvertRotationMatrix(const Eigen::Matrix3f& rotationMatrix)
	{
		glm::mat3 result = glm::mat3();
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				result[i][j] = rotationMatrix(j, i);
			}
		}
		return result;
	}

	glm::vec3 ConvertTranslationVector(const Eigen::Vector3f& translationVector)
	{
		return glm::vec3(translationVector[0], translationVector[1], translationVector[2]);
	}

	void PrintMatrix(Eigen::Matrix3f matrix)
	{
		std::cout << matrix << std::endl;
	}

	void PrintMatrixWithSize(const glm::mat4& matrix, int size)
	{
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++)
				printf("%1.8f ", matrix[i][j]);

			printf("\n");
		}
	}

	void PrintMatrix(const glm::mat4& matrix)
	{
		PrintMatrixWithSize(matrix, 4);
	}

	void PrintMatrix(const glm::mat3& matrix)
	{
		PrintMatrixWithSize(matrix, 3);
	}
}