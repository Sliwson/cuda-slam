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
			return loader.GetMergedCloud();
		else
			return std::vector<Point_f>();
	}

	std::vector<Point_f> GetSubcloud(const std::vector<Point_f>& cloud, const std::vector<int>& indices)
	{
		if (indices.size() >= cloud.size())
			return cloud;

		std::vector<Point_f> subcloud(indices.size());
		std::transform(indices.begin(), indices.end(), subcloud.begin(), [&cloud](size_t pos) { return cloud[pos]; });

		return subcloud;
	}

	std::vector<Point_f> ResizeCloudWithStep(const std::vector<Point_f>& cloud, int step)
	{
		int size = cloud.size() / step;
		if (size == 0)
			return cloud;
		std::vector<Point_f> result = std::vector<Point_f>(size);
		size_t i = 0, j = 0;
		for (i = 0, j = 0; i < cloud.size(); i += step, j++);
		{
			result[j] = cloud[i];
		}
		return result;
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

	Point_f TransformPoint(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& scale)
	{
		const glm::vec3 result = scale * (rotationMatrix * point) + translationVector;
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

	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& scale)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](const Point_f& p) { return TransformPoint(p, rotationMatrix, translationVector, scale); });
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

	float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		float diffSum = 0.0f;
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto diff = cloudAfter[i] - cloudBefore[i];
			diffSum += diff.LengthSquared();
		}
		return diffSum / cloudBefore.size();
	}

	//float GetMeanSquaredError(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& correspondingIndexes)
	//{
	//	float diffSum = 0.0f;
	//	for (int i = 0; i < correspondingIndexes.size(); i++)
	//	{
	//		const auto diff = cloudAfter[i] - cloudBefore[correspondingIndexes[i]];
	//		diffSum += diff.LengthSquared();
	//	}
	//	return diffSum / correspondingIndexes.size();
	//}

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

	Eigen::VectorXf GetVectorXFromPointsVector(const std::vector<float>& vector)
	{
		Eigen::VectorXf result = Eigen::VectorXf::Zero(vector.size());
		for (int i = 0; i < vector.size(); i++)
		{
			result(i) = vector[i];
		}
		return result;
	}

	Eigen::MatrixXf GetMatrixXFromPointsVector(const std::vector<float>& points, const int& rows, const int& cols)
	{
		Eigen::MatrixXf result = Eigen::ArrayXXf::Zero(rows, cols);
		for (size_t x = 0; x < rows; x++)
		{
			for (size_t y = 0; y < cols; y++)
			{
				result(x, y) = points[x * cols + y];
			}
		}
		return result;
	}

	Eigen::Vector3f ConvertToEigenVector(const Point_f& point)
	{
		return Eigen::Vector3f(point.x, point.y, point.z);
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

	glm::mat4 ConvertToTransformationMatrix(const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		auto matrix = glm::mat4(rotationMatrix);
		matrix[3] = glm::vec4(translationVector, 1.0f);
		return matrix;
	}

	void PrintMatrix(Eigen::Matrix3f matrix)
	{
		std::stringstream ss;
		ss << matrix;
		printf("%s\n", ss.str().c_str());
	}

	void PrintMatrixWithSize(const glm::mat4& matrix, int size)
	{
		for (int j = 0; j < size; j++) {
			for (int i = 0; i < size; i++)
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

	void PrintMatrix(const glm::mat3& matrix, const glm::vec3& vector)
	{
		const auto transform = ConvertToTransformationMatrix(matrix, vector);
		PrintMatrix(transform);
	}

	CorrespondingPointsTuple GetCorrespondingPoints(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared)
	{
		std::vector<Point_f> correspondingFromCloudBefore(cloudBefore.size());
		std::vector<Point_f> correspondingFromCloudAfter(cloudBefore.size());
		std::vector<int> correspondingIndexesBefore(cloudBefore.size());
		std::vector<int> correspondingIndexesAfter(cloudBefore.size());
		int correspondingCount = 0;

		for (int i = 0; i < cloudBefore.size(); i++)
		{
			int closestIndex = -1;
			float closestDistance = std::numeric_limits<float>::max();

			for (int j = 0; j < cloudAfter.size(); j++)
			{
				float distance = (cloudAfter[j] - cloudBefore[i]).LengthSquared();

				if (distance < closestDistance || closestIndex == -1)
				{
					closestDistance = distance;
					closestIndex = j;
				}
			}
			if (closestDistance < maxDistanceSquared && closestIndex >= 0)
			{
				correspondingFromCloudBefore[correspondingCount] = cloudBefore[i];
				correspondingFromCloudAfter[correspondingCount] = cloudAfter[closestIndex];
				correspondingIndexesBefore[correspondingCount] = i;
				correspondingIndexesAfter[correspondingCount] = closestIndex;

				correspondingCount++;
			}
		}

		correspondingFromCloudBefore.resize(correspondingCount);
		correspondingFromCloudAfter.resize(correspondingCount);
		correspondingIndexesBefore.resize(correspondingCount);
		correspondingIndexesAfter.resize(correspondingCount);

		return std::make_tuple(correspondingFromCloudBefore, correspondingFromCloudAfter, correspondingIndexesBefore, correspondingIndexesAfter);
	}

	std::pair<glm::mat3, glm::vec3> LeastSquaresSVD(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		Eigen::Matrix3f rotationMatrix;// = Eigen::Matrix3f::Identity();
		Eigen::Vector3f translationVector;// = Eigen::Vector3f::Zero();

		auto centerBefore = GetCenterOfMass(cloudBefore);
		auto centerAfter = GetCenterOfMass(cloudAfter);

		const Eigen::Matrix3Xf alignedBefore = GetMatrix3XFromPointsVector(GetAlignedCloud(cloudBefore, centerBefore));
		const Eigen::Matrix3Xf alignedAfter = GetMatrix3XFromPointsVector(GetAlignedCloud(cloudAfter, centerAfter));

		//version with thinU (docs say it should be enough for us, further efficiency tests are needed
		// cheching option with MatrixXf instead od Matrix3f because documantation says: my matrix should have dynamic number of columns
		const Eigen::MatrixXf matrix = alignedAfter * alignedBefore.transpose();
		const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		//version with fullU
		//const Eigen::Matrix3f matrix = alignedAfter * alignedBefore.transpose();
		//const Eigen::JacobiSVD<Eigen::Matrix3f> svd = Eigen::JacobiSVD<Eigen::Matrix3f>(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVtransposed = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVtransposed;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant());

		rotationMatrix = matrixU * diag * matrixVtransposed;

		glm::mat3 rotationMatrixGLM = ConvertRotationMatrix(rotationMatrix);

		glm::vec3 translationVectorGLM = glm::vec3(centerAfter) - (rotationMatrixGLM * centerBefore);

		return std::make_pair(rotationMatrixGLM, translationVectorGLM);
	}

	std::vector<int> GetRandomPermutationVector(int size)
	{
		std::vector<int> permutation(size);
		std::iota(permutation.begin(), permutation.end(), 0);
		std::shuffle(permutation.begin(), permutation.end(), std::mt19937{ std::random_device{}() });
		return permutation;
	}

	std::vector<int> InversePermutation(const std::vector<int>& permutation)
	{
		auto inversedPermutation = std::vector<int>(permutation.size());
		for (int i = 0; i < permutation.size(); i++)
		{
			inversedPermutation[permutation[i]] = i;
		}
		return inversedPermutation;
	}

	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation)
	{
		std::vector<Point_f> permutedCloud(input.size());
		for (int i = 0; i < input.size(); i++)
			permutedCloud[i] = input[permutation[i]];
		return permutedCloud;
	}
}