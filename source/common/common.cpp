#include <chrono>
#include <tuple>
#include <thread>
#include <array>

#include "testutils.h"
#include "common.h"
#include "configuration.h"
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

	std::vector<Point_f> GetSubcloud(const std::vector<Point_f>& cloud, int subcloudSize)
	{
		if (subcloudSize >= cloud.size())
			return cloud;

		std::vector<int> subcloudIndices = GetRandomPermutationVector(static_cast<int>(cloud.size()));
		subcloudIndices.resize(subcloudSize);

		std::vector<Point_f> subcloud(subcloudIndices.size());
		std::transform(subcloudIndices.begin(), subcloudIndices.end(), subcloud.begin(), [&cloud](size_t pos) { return cloud[pos]; });

		return subcloud;
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

	std::vector<Point_f> NormalizeCloud(const std::vector<Point_f>& cloud, float size)
	{
		const auto massCenter = GetCenterOfMass(cloud);
		auto alignedCloud = GetAlignedCloud(cloud, massCenter);

		const auto get_minmax = [&](auto selector) {
			 return std::minmax_element(alignedCloud.rbegin(), alignedCloud.rend(), [selector](const auto& p1, const auto& p2) { return selector(p1) < selector(p2); });
		};

		const auto [xMin, xMax] = get_minmax([](auto p) { return p.x; });
		const auto [yMin, yMax] = get_minmax([](auto p) { return p.y; });
		const auto [zMin, zMax] = get_minmax([](auto p) { return p.z; });

		const std::array<float, 3> spans = { xMax->x - xMin->x, yMax->y - yMin->y, zMax->z - zMin->z };
		const auto max = std::max_element(spans.begin(), spans.end());

		if (*max == 0)
			return cloud;

		const auto scale = size / *max;

		std::transform(alignedCloud.begin(), alignedCloud.end(), alignedCloud.begin(), [scale](auto p) { return p * scale; });
		return GetAlignedCloud(alignedCloud, massCenter * -1.f);
	}

	std::pair<std::vector<Point_f>, std::vector<Point_f>> GetCloudsFromConfig(Configuration config)
	{
		const auto sameClouds = config.BeforePath == config.AfterPath;

		auto before = LoadCloud(config.BeforePath);
		auto after = sameClouds ? before : LoadCloud(config.AfterPath);

		// scale clouds if necessary
		if (config.CloudResize.has_value())
		{
			const auto newSize = config.CloudResize.value();
			before = GetSubcloud(before, newSize);
			after = sameClouds ? before : GetSubcloud(after, newSize);
		}

		// normalize clouds to standart value
		before = NormalizeCloud(before, CLOUD_BOUNDARY);
		after = NormalizeCloud(after, CLOUD_BOUNDARY);


		// shuffle clouds
		std::shuffle(before.begin(), before.end(), std::mt19937{ std::random_device{}() });
		std::shuffle(after.begin(), after.end(), std::mt19937{ std::random_device{}() });

		// apply transformation and return
		if (config.Transformation.has_value())
		{
			const auto& transform = config.Transformation.value();
			return std::make_pair(
				before,
				GetTransformedCloud(after, transform.first, transform.second)
			);
		}
		else if (config.TransformationParameters.has_value())
		{
			const auto params = config.TransformationParameters.value();
			const auto rotationVal = params.first;
			const auto translationVal = params.second;
			const auto rotation = Tests::GetRandomRotationMatrix(rotationVal);
			const auto translation = Tests::GetRandomTranslationVector(translationVal);

			return std::make_pair(
				before,
				GetTransformedCloud(after, rotation, translation)
			);
		}
		else
		{
			assert(false); // Wrong configuration!
			return std::make_pair(before, after);
		}
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

	std::pair<glm::mat3, glm::vec3> ConvertToRotationTranslationPair(const glm::mat4& transformationMatrix)
	{
		auto matrix = glm::mat3(transformationMatrix);
		auto vector = glm::vec3(transformationMatrix[3]);
		return std::make_pair(matrix, vector);
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

	CorrespondingPointsTuple GetCorrespondingPointsSequential(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared)
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

	CorrespondingPointsTuple GetCorrespondingPointsParallel(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared)
	{
		const auto threadCount = std::thread::hardware_concurrency();
		std::vector<std::thread> workerThreads;

		const auto get_correspondence_idx = [](Point_f sourcePoint, const std::vector<Point_f>& targetCloud) {
			int closestIndex = -1;
			float closestDistance = std::numeric_limits<float>::max();

			for (int j = 0; j < targetCloud.size(); j++)
			{
				float distance = (targetCloud[j] - sourcePoint).LengthSquared();

				if (distance < closestDistance)
				{
					closestDistance = distance;
					closestIndex = j;
				}
			}

			return closestIndex;
		};

		std::vector<int> correspondingIndices(cloudBefore.size());

		const auto calculate_correspondences = [&](int beginIndex, int endIndex) {
			for (int i = beginIndex; i < endIndex; i++)
				correspondingIndices[i] = get_correspondence_idx(cloudBefore[i], cloudAfter);
		};

		const auto threadWorkLength = cloudBefore.size() / threadCount;
		for (unsigned int i = 0; i < threadCount - 1; i++)
			workerThreads.push_back(std::thread(calculate_correspondences, i * threadWorkLength, (i + 1) * threadWorkLength));

		workerThreads.push_back(std::thread(calculate_correspondences, (threadCount - 1) * threadWorkLength, cloudBefore.size()));

		for (unsigned int i = 0; i < threadCount; i++)
			workerThreads[i].join();

		std::vector<Point_f> correspondingFromCloudBefore(cloudBefore.size());
		std::vector<Point_f> correspondingFromCloudAfter(cloudBefore.size());
		std::vector<int> correspondingIndexesBefore(cloudBefore.size());
		std::vector<int> correspondingIndexesAfter(cloudBefore.size());
		int correspondingCount = 0;

		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto closestIndex = correspondingIndices[i];
			const auto distance = (cloudBefore[i] - cloudAfter[closestIndex]).LengthSquared();
			if (distance < maxDistanceSquared)
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

	CorrespondingPointsTuple GetCorrespondingPoints(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared, bool parallel)
	{
		if (parallel)
			return GetCorrespondingPointsParallel(cloudBefore, cloudAfter, maxDistanceSquared);
		else
			return GetCorrespondingPointsSequential(cloudBefore, cloudAfter, maxDistanceSquared);
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
			permutedCloud[i] = i < permutation.size() ? input[permutation[i]] : input[i];

		return permutedCloud;
	}
}
