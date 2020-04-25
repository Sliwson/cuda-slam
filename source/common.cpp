#include <Eigen/Dense>
#include <chrono>
#include <tuple>

#include "common.h"
#include "loader.h"

namespace Common
{

	
	namespace
	{
		constexpr float TEST_EPS = 1e-4f;

		float GetRandomFloat(float min, float max)
		{
			float range = max - min;
			return static_cast<float>(rand()) / RAND_MAX * range + min;
		}

		Point_f GetRandomPoint(const Point_f& min, const Point_f& max)
		{
			return {
				GetRandomFloat(min.x, max.x),
				GetRandomFloat(min.y, max.y),
				GetRandomFloat(min.z, max.z)
			};
		}

		Point_f TransformPoint(const Point_f& in, const glm::mat4& matrix)
		{
			const glm::vec3 result = matrix * glm::vec4(glm::vec3(in), 1.0f);
			return Point_f(result);
		}
	}

	std::tuple<std::vector<Common::Point_f>, Eigen::Matrix3f, Eigen::Vector3f> SimpleSVD(std::vector<Common::Point_f>& before, std::vector<Common::Point_f>& after);
	int testfunc();

	std::vector<Point_f> LoadCloud(const std::string& path)
	{
		AssimpCloudLoader loader(path);
		if (loader.GetCloudCount() > 0)
			return loader.GetCloud(0);
		else
			return std::vector<Point_f>();
	}

	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count)
	{
		std::vector<Point_f> result;
		for (int i = 0; i < count; i++)
			result.push_back(GetRandomPoint(corner, corner + size));

		return result;
	}

	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRange)
	{
		const auto rotation = glm::rotate(glm::mat4(1.0f),
			rotationRange,//GetRandomFloat(0, rotationRange),
			glm::vec3(GetRandomPoint(Point_f::Zero(), Point_f::One())));

		return glm::translate(rotation, glm::vec3(GetRandomPoint(translationMin, translationMax)));
	}

	std::vector<Point_f> GetTransformedCloud(const std::vector<Point_f>& cloud, const glm::mat4& matrix)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](Point_f p) { return TransformPoint(p, matrix); });
		return clone;
	}

	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation)
	{
		std::vector<Point_f> permutedCloud(input.size());
		for (int i = 0; i < input.size(); i++)
			permutedCloud[i] = input[permutation[i]];

		return permutedCloud;
	}

	Point_f GetCenterOfMass(const std::vector<Point_f>& cloud)
	{
		return std::accumulate(cloud.begin(), cloud.end(), Point_f::Zero()) / (float)cloud.size();
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

	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, matrix) <= TEST_EPS;
	}

	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, matrix);
	}

	bool TestPermutation(const std::vector<int>& expected, const std::vector<int>& actual)
	{
		return actual == expected;
	}

	std::vector<int> GetRandomPermutationVector(int size)
	{
		std::vector<int> permutation(size);
		std::iota(permutation.begin(), permutation.end(), 0);
		std::shuffle(permutation.begin(), permutation.end(), std::mt19937{ std::random_device{}() });
		return permutation;
	}

	std::vector<int> GetClosestPointIndexes(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		const auto size = cloudBefore.size();
		std::vector<int> resultPermutation(size);

		return resultPermutation;
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

	std::pair<std::vector<Point_f>, std::vector<Point_f>>GetCorrespondingPoints(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, float maxDistanceSquared)
	{
		std::vector<Point_f> correspondingFromCloudBefore(cloudBefore.size());
		std::vector<Point_f> correspondingFromCloudAfter(cloudBefore.size());
		int correspondingCount = 0;

		for (int i = 0; i < cloudBefore.size(); i++)
		{
			int closestIndex = -1;
			int closestDistance = 0;

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

				correspondingCount++;
			}
		}

		correspondingFromCloudBefore.resize(correspondingCount);
		correspondingFromCloudAfter.resize(correspondingCount);

		return std::make_pair(correspondingFromCloudBefore, correspondingFromCloudAfter);
	}

	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud)
	{
		auto center = GetCenterOfMass(cloud);
		auto result = std::vector<Point_f>(cloud.size());
		// TODO: Use thrust maybe and parametrize host/device run
		std::transform(cloud.begin(), cloud.end(), result.begin(),
			[center](const Point_f& point) -> Point_f { return point - center; });

		return result;
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

	glm::mat4 LeastSquaresSVD(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& orderedCloudAfter, float* error)
	{
		auto transformationMatrix = glm::mat4();

		auto centerBefore = GetCenterOfMass(cloudBefore);
		auto centerAfter = GetCenterOfMass(orderedCloudAfter);

		Eigen::Matrix3Xf alignedBefore = GetMatrix3XFromPointsVector(GetAlignedCloud(cloudBefore));
		Eigen::Matrix3Xf alignedAfter = GetMatrix3XFromPointsVector(GetAlignedCloud(orderedCloudAfter));

		Eigen::Matrix3f matrix = alignedAfter * alignedBefore.transpose();

		//// Official documentation says thin U and thin V are enough for us, not gonna argue
		//// But maybe it is not enough

		//TODO: check if full u doesnt need Xf matrix
		Eigen::JacobiSVD<Eigen::Matrix3f> const svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		double det = (svd.matrixU() * svd.matrixV().transpose()).determinant();
		Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, det);
		Eigen::Matrix3f rotation = svd.matrixU() * diag * svd.matrixV().transpose();

		Point_f translation = centerAfter - (rotation * centerBefore);

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				transformationMatrix[i][j] = rotation(j, i);
			}
			transformationMatrix[i][3] = 0;
		}

		transformationMatrix[3][0] = translation.x;
		transformationMatrix[3][1] = translation.y;
		transformationMatrix[3][2] = translation.z;
		transformationMatrix[3][3] = 1.0f;

		return transformationMatrix;
	}

	glm::mat4 GetTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float maxDistanceSquared, int maxIterations = -1)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat4 transformationMatrix(1.0f);
		std::pair<std::vector<Point_f>, std::vector<Point_f>> closestPoints;

		//Eigen::Matrix3Xf cloudBeforeMatrix = GetMatrix3XFromPointsVector(cloudBefore);
		//Eigen::Matrix3Xf cloudAfterMatrix = GetMatrix3XFromPointsVector(cloudAfter);

		while (maxIterations == -1 || *iterations < maxIterations)
		{
			//printf("Obrot %d\n", *iterations);
			// 1. difference: Maybe use cloud in Matrix here?
			auto transformedCloudBefore = GetTransformedCloud(cloudBefore, transformationMatrix);

			auto correspondingPoints = GetCorrespondingPoints(transformedCloudBefore, cloudAfter, maxDistanceSquared);
			if (correspondingPoints.first.size() == 0)
				break;

			// Here we multiply
			transformationMatrix = LeastSquaresSVD(correspondingPoints.first, correspondingPoints.second, error) * transformationMatrix;

			*error = GetMeanSquaredError(correspondingPoints.first, correspondingPoints.second, transformationMatrix);
			//powinnismy wziac corresponding points first i second

			if (*error < TEST_EPS)
				break;

			(*iterations)++;
		}

		return transformationMatrix;
	}

	//refactored functions-------------------------------------------------------------------------------------------------------

	Eigen::Matrix3Xf GetMatrix3XFromPointsVector_Refactor(const std::vector<Point_f>& points)
	{
		Eigen::Matrix3Xf result(3, points.size());
		for (int i = 0; i < points.size(); i++)
		{
			result.col(i) = static_cast<Eigen::Vector3f>(points[i]);
		}

		return result;
	}

	Eigen::Matrix3Xf GetTransformedCloud_Refactor(const Eigen::Matrix3Xf& cloud, const Eigen::Matrix3f& rotationMatrix, const Eigen::Vector3f& translationVector)
	{
		Eigen::Matrix3Xf result = rotationMatrix * cloud;
		result.colwise() += translationVector;
		return result;
	}

	float CalculateSquareDistance(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < v1.rows(); i++)
		{
			sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		}
		return sum;
	}

	std::pair<Eigen::Matrix3Xf, Eigen::Matrix3Xf>GetCorrespondingPoints_Refactor(const Eigen::Matrix3Xf& cloudBefore, const Eigen::Matrix3Xf& cloudAfter, float maxDistanceSquared)
	{
		Eigen::Matrix3Xf correspondingFromCloudBefore = Eigen::Array3Xf::Zero(3, cloudBefore.cols());
		Eigen::Matrix3Xf correspondingFromCloudAfter(3, cloudAfter.cols());
		int correspondingCount = 0;

		for (int i = 0; i < cloudBefore.cols(); i++)
		{
			int closestIndex = -1;
			int closestDistance = 0;

			for (int j = 0; j < cloudAfter.cols(); j++)
			{
				float distance = CalculateSquareDistance(cloudAfter.col(j), cloudBefore.col(i));

				if (distance < closestDistance || closestIndex == -1)
				{
					closestDistance = distance;
					closestIndex = j;
				}
			}
			if (closestDistance < maxDistanceSquared && closestIndex >= 0)
			{
				correspondingFromCloudBefore.col(correspondingCount) = cloudBefore.col(i);
				correspondingFromCloudAfter.col(correspondingCount) = cloudAfter.col(closestIndex);

				correspondingCount++;
			}
		}

		correspondingFromCloudBefore.conservativeResize(Eigen::NoChange_t::NoChange, correspondingCount);
		correspondingFromCloudAfter.conservativeResize(Eigen::NoChange_t::NoChange, correspondingCount);

		return std::make_pair(correspondingFromCloudBefore, correspondingFromCloudAfter);
	}

	Eigen::Vector3f GetCenterOfMass_Refactor(const Eigen::Matrix3Xf& cloud)
	{
		int count = cloud.cols();
		Eigen::Vector3f result = cloud * Eigen::VectorXf::Ones(count);
		result /= static_cast<float>(count);
		return result;
	}

	Eigen::Matrix3Xf GetAlignedCloud_Refactor(const Eigen::Matrix3Xf& cloud, const Eigen::Vector3f& center_mass)
	{
		Eigen::Matrix3Xf result = cloud;
		result.colwise() -= center_mass;
		return result;
	}

	std::pair<Eigen::Matrix3f, Eigen::Vector3f> LeastSquaresSVD_Refactor(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		Eigen::Matrix3f rotationMatrix;// = Eigen::Matrix3f::Identity();
		Eigen::Vector3f translationVector;// = Eigen::Vector3f::Zero();

		const Eigen::Matrix3Xf cloudBeforeMatrix = GetMatrix3XFromPointsVector_Refactor(cloudBefore);
		const Eigen::Matrix3Xf cloudAfterMatrix = GetMatrix3XFromPointsVector_Refactor(cloudAfter);

		const Eigen::Vector3f centerBefore = GetCenterOfMass_Refactor(cloudBeforeMatrix);
		const Eigen::Vector3f centerAfter = GetCenterOfMass_Refactor(cloudAfterMatrix);

		const Eigen::Matrix3Xf alignedBefore = GetAlignedCloud_Refactor(cloudBeforeMatrix, centerBefore);
		const Eigen::Matrix3Xf alignedAfter = GetAlignedCloud_Refactor(cloudAfterMatrix, centerAfter);

		const Eigen::MatrixXf matrix = alignedAfter * alignedBefore.transpose();

		//// Official documentation says thin U and thin V are enough for us, not gonna argue
		//// But maybe it is not enough
		Eigen::JacobiSVD<Eigen::MatrixXf> const svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVtransposed = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVtransposed;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant());

		rotationMatrix = matrixU * diag * matrixVtransposed;

		translationVector = centerAfter - (rotationMatrix * centerBefore);
		
		return std::make_pair(rotationMatrix, translationVector);
	}

	Point_f TransformPoint_Refactor(const Point_f& point, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		const glm::vec3 result = (rotationMatrix * point) + translationVector;
		return Point_f(result);
	}

	float GetMeanSquaredError_Refactor(const Eigen::Matrix3Xf& cloudBefore, const Eigen::Matrix3Xf& cloudAfter)
	{
		const Eigen::Matrix3Xf diffMatrix = cloudBefore - cloudAfter;
		return diffMatrix.squaredNorm();// maybe we should /diffMatrix.cols()
	}

	float GetMeanSquaredError_Refactor(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		float diffSum = 0.0f;
		// We assume clouds are the same size but if error is significant, you might want to check it
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto transformed = TransformPoint_Refactor(cloudBefore[i], rotationMatrix, translationVector);
			const auto diff = cloudAfter[i] - transformed;
			diffSum += diff.LengthSquared();
		}

		return diffSum; // / cloudBefore.size();
	}

	float GetMeanSquaredError_Refactor(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		float diffSum = 0.0f;
		// We assume clouds are the same size but if error is significant, you might want to check it
		for (int i = 0; i < cloudBefore.size(); i++)
		{
			const auto diff = cloudAfter[i] - cloudBefore[i];
			diffSum += diff.LengthSquared();
		}

		return diffSum; // / cloudBefore.size();
	}

	glm::mat3 ConvertRotationMatrix(const Eigen::Matrix3f& rotationMatrix)
	{
		glm::mat3 result = glm::mat3();
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				//szymon has rotationMatrix(j,i)
				result[i][j] = rotationMatrix(j, i);
			}
		}
		return result;
	}

	glm::vec3 ConvertTranslationVector(const Eigen::Vector3f& translationVector)
	{
		return glm::vec3(translationVector[0], translationVector[1], translationVector[2]);
	}

	std::vector<Point_f> GetTransformedCloud_Refactor(const std::vector<Point_f>& cloud, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		auto clone = cloud;
		std::transform(clone.begin(), clone.end(), clone.begin(), [&](Point_f p) { return TransformPoint_Refactor(p, rotationMatrix, translationVector); });
		return clone;
	}

	bool TestTransformOrdered_Refactor(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError_Refactor(cloudBefore, cloudAfter, rotationMatrix, translationVector) <= TEST_EPS;
	}

	bool TestTransformWithPermutation_Refactor(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered_Refactor(permutedCloudBefore, cloudAfter, rotationMatrix, translationVector);
	}

	std::pair<glm::mat3, glm::vec3> BasicICP(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* iterations, float* error, float maxDistanceSquared, int maxIterations = -1)
	{
		*iterations = 0;
		*error = 1e5;
		Eigen::Matrix3f rotationMatrix = Eigen::Matrix3f::Identity();
		Eigen::Vector3f translationVector = Eigen::Vector3f::Zero();
		std::pair<std::vector<Point_f>, std::vector<Point_f>> closestPoints;

		while (maxIterations == -1 || *iterations < maxIterations)
		{
			//get corresponding points
			auto transformedCloud = GetTransformedCloud_Refactor(cloudBefore, ConvertRotationMatrix(rotationMatrix), ConvertTranslationVector(translationVector));

			auto correspondingPoints = GetCorrespondingPoints(transformedCloud, cloudAfter, maxDistanceSquared);
			if (correspondingPoints.first.size() == 0)
				break;

			// use svd
			auto transformationMatrix = LeastSquaresSVD_Refactor(correspondingPoints.first, correspondingPoints.second);

			// not sure if we should do this this way but it should work
			rotationMatrix = transformationMatrix.first * rotationMatrix;
			translationVector = translationVector + transformationMatrix.second;

			glm::mat3 currentRotationMatrix = ConvertRotationMatrix(transformationMatrix.first);
			glm::vec3 currentTranslationVector = ConvertTranslationVector(transformationMatrix.second);

			// count error
			*error = GetMeanSquaredError_Refactor(correspondingPoints.first, correspondingPoints.second, currentRotationMatrix, currentTranslationVector);
			//powinnismy wziac corresponding points first i second

			//printf("Obrot %d, error: %f\n", *iterations, *error);

			if (*error < TEST_EPS)
			{
				break;
			}

			(*iterations)++;
		}
		//while (maxIterations == -1 || *iterations < maxIterations)
		//{
		//	//get corresponding points
		//	auto transformedCloud = GetTransformedCloud_Refactor(cloudBefore, ConvertRotationMatrix(rotationMatrix), ConvertTranslationVector(translationVector));

		//	auto correspondingPoints = GetCorrespondingPoints(transformedCloud, cloudAfter, maxDistanceSquared);
		//	if (correspondingPoints.first.size() == 0)
		//		break;

		//	// use svd
		//	auto transformationMatrix = SimpleSVD(correspondingPoints.first, correspondingPoints.second);

		//	// not sure if we should do this this way but it should work
		//	rotationMatrix = std::get<1>(transformationMatrix) * rotationMatrix;
		//	translationVector = translationVector + std::get<2>(transformationMatrix);

		//	glm::mat3 currentRotationMatrix = ConvertRotationMatrix(std::get<1>(transformationMatrix));
		//	glm::vec3 currentTranslationVector = ConvertTranslationVector(std::get<2>(transformationMatrix));

		//	float comparision = GetMeanSquaredError_Refactor(std::get<0>(transformationMatrix), correspondingPoints.second);
		//	std::cout << "compare: " << comparision << std::endl;

		//	// count error
		//	*error = GetMeanSquaredError_Refactor(correspondingPoints.first, correspondingPoints.second, currentRotationMatrix, currentTranslationVector);
		//	//powinnismy wziac corresponding points first i second

		//	printf("Obrot %d, error: %f\n", *iterations, *error);

		//	if (*error < TEST_EPS)
		//	{
		//		break;
		//	}

		//	(*iterations)++;
		//}
		return std::make_pair(ConvertRotationMatrix(rotationMatrix), ConvertTranslationVector(translationVector));
	}

	void PrintMatrix(glm::mat3 matrix, glm::vec3 vec3)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				std::cout << matrix[j][i] << '\t';
			}
			std::cout << vec3[i];
			std::cout << std::endl;
		}
		std::cout << "0\t0\t0\t1\t" << std::endl;
	}

	

	void PrintMatrix(glm::mat4 matrix)
	{
		for (size_t i = 0; i < 4; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				std::cout << matrix[j][i] << '\t';
			}
			std::cout << std::endl;
		}
	}


	//refactor end -----------------------------------------------------------------------------------------------------------

	void LibraryTest()
	{
		srand(666);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		int iterations = 0;
		float error = 1.0f;

		//const auto cloud = GetRandomPointCloud(corner, size, 3000);
		auto cloud = LoadCloud("data/bunny.obj");

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		cloud.resize(3000);
		int cloudSize = cloud.size();

		//old tests
		if(true)
		{
			std::cout << "Szymon" << std::endl;
			const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
			const auto permutation = GetRandomPermutationVector(cloudSize);
			const auto permutedCloud = ApplyPermutation(cloud, permutation);
			const auto transformedCloud = GetTransformedCloud(cloud, transform);
			const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

			const auto calculatedPermutation = InversePermutation(GetClosestPointIndexes(cloud, transformedPermutedCloud));
			auto icp1start = std::chrono::high_resolution_clock::now();
			const auto icpCalculatedTransform1 = GetTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, 25.0f, 5);
			auto icp2start = std::chrono::high_resolution_clock::now();
			const auto icpCalculatedTransform2 = GetTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, 25.0f, 50);
			auto icp2end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> icp1duration = icp2start - icp1start;
			std::chrono::duration<double> icp2duration = icp2end - icp2start;

			const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform);
			const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform);
			const auto resultPermutation = TestPermutation(permutation, calculatedPermutation);

			printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
			printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
			printf("Permutation find test [%s]\n", resultPermutation ? "OK" : "FAIL");
			printf("ICP test (%d iterations) error = %g\n", iterations, error);
			printf("ICP test 1 duration %f\n", icp1duration.count());
			printf("ICP test 2 duration %f\n", icp2duration.count());

			std::cout << "Transform Matrix" << std::endl;
			PrintMatrix(transform);

			std::cout << "ICP2 Matrix" << std::endl;
			PrintMatrix(icpCalculatedTransform2);

			Common::Renderer renderer(
				Common::ShaderType::SimpleModel,
				cloud, //grey
				transformedCloud, //blue
				GetTransformedCloud(cloud, icpCalculatedTransform1), //red
				GetTransformedCloud(cloud, icpCalculatedTransform2)); //green
			
			renderer.Show();
		  
		}

		//new tests
		if(false)
		{
			std::cout << "Michal" << std::endl;
			const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
			const auto permutation = GetRandomPermutationVector(cloudSize);
			const auto permutedCloud = ApplyPermutation(cloud, permutation);
			const auto transformedCloud = GetTransformedCloud(cloud, transform);
			const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

			const auto calculatedPermutation = InversePermutation(GetClosestPointIndexes(cloud, transformedPermutedCloud));
			auto icp1start = std::chrono::high_resolution_clock::now();
			const auto icpCalculatedTransform1 = BasicICP(cloud, transformedPermutedCloud, &iterations, &error, 25.0f, 5);
			auto icp2start = std::chrono::high_resolution_clock::now();
			iterations = 0;
			error = 1.0f;
			const auto icpCalculatedTransform2 = BasicICP(cloud, transformedPermutedCloud, &iterations, &error, 25.0f, 50);
			auto icp2end = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double> icp1duration = icp2start - icp1start;
			std::chrono::duration<double> icp2duration = icp2end - icp2start;

			const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform);
			const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform);
			const auto resultPermutation = TestPermutation(permutation, calculatedPermutation);

			printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
			printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
			printf("Permutation find test [%s]\n", resultPermutation ? "OK" : "FAIL");
			printf("ICP test (%d iterations) error = %g\n", iterations, error);
			printf("ICP test 1 duration %f\n", icp1duration.count());
			printf("ICP test 2 duration %f\n", icp2duration.count());


			std::cout << "Transform Matrix" << std::endl;
			PrintMatrix(transform);

			std::cout << "ICP2 Matrix" << std::endl;
			PrintMatrix(icpCalculatedTransform2.first, icpCalculatedTransform2.second);

			Common::Renderer renderer(
				Common::ShaderType::SimpleModel,
				cloud, //grey
				transformedCloud, //blue
				GetTransformedCloud_Refactor(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //red
				GetTransformedCloud_Refactor(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //green

			renderer.Show();
		}
	}



int testfunc()
{
	printf("Hello cpu-slam!\n");
	//Common::LibraryTest();

	auto cloud = LoadCloud("data/bunny.obj");
	//std::vector<Point_f> cloud = std::vector<Point_f>(1500);
	//for (size_t i = 0; i < cloud.size(); i++)
	//{
	//	cloud[i] = Point_f(i, i, i);
	//}

	printf("Cloud size: %d\n", cloud.size());

	Eigen::Matrix<float, 3, 3> rotationMatrix;
	float angle = 45.0f;

	rotationMatrix << 1.0f, 0.0f, 0.0f,
		0.0f, std::cos(angle), -std::sin(angle),
		0, std::sin(angle), std::cos(angle);

	Eigen::Matrix<float, 3, Eigen::Dynamic> beforeMatrix(3, cloud.size());
	Eigen::Vector3f translationVector(1.0f, 0.0f, 0.0f);
	for (long i = 0; i < cloud.size(); i++)
	{
		beforeMatrix(0, i) = cloud[i].x;
		beforeMatrix(1, i) = cloud[i].y;
		beforeMatrix(2, i) = cloud[i].z;
	}

	Eigen::Matrix<float, 3, Eigen::Dynamic> rotatedCloud = rotationMatrix * beforeMatrix;

	std::vector<Point_f> cloud_after = std::vector<Point_f>(cloud.size());

	for (long i = 0; i < cloud.size(); i++)
	{
		cloud_after[i].x = rotatedCloud(0, i) - translationVector(0);
		cloud_after[i].y = rotatedCloud(1, i) - translationVector(1);
		cloud_after[i].z = rotatedCloud(2, i) - translationVector(2);
	}

	auto cpu_result = SimpleSVD(cloud, cloud_after);

	//Renderer renderer(ShaderType::SimpleModel, cloud, cloud_after, std::get<0>(cpu_result), std::vector<Point_f>(1));

	//renderer.Show();

	float comparision = GetMeanSquaredError_Refactor(std::get<0>(cpu_result), cloud_after);
	std::cout << "compare: " << comparision << std::endl;

	return 0;
}

std::tuple<std::vector<Common::Point_f>, Eigen::Matrix3f, Eigen::Vector3f> SimpleSVD(std::vector<Common::Point_f>& before, std::vector<Common::Point_f>& after)
{
	//Common::Point_f sum_before = Common::Point_f::Zero();
	//Common::Point_f sum_after = Common::Point_f::Zero();
	//std::for_each(before.begin(), before.end(), [&sum_before](const Common::Point_f& el) {sum_before = sum_before + el; });
	//std::for_each(after.begin(), after.end(), [&sum_after](const Common::Point_f& el) {sum_after = sum_after + el; });
	Common::Point_f sum_before = std::accumulate(before.begin(), before.end(), Common::Point_f::Zero());
	Common::Point_f sum_after = std::accumulate(after.begin(), after.end(), Common::Point_f::Zero());
	Common::Point_f before_centre = sum_before / (float)before.size();
	Common::Point_f after_centre = sum_after / (float)after.size();

	std::vector<Common::Point_f> before_transformed = std::vector<Common::Point_f>(before.size());
	std::vector<Common::Point_f> after_transformed = std::vector<Common::Point_f>(after.size());

	std::transform(before.begin(), before.end(), before_transformed.begin(), [&before_centre](const Common::Point_f& el) {return el - before_centre; });
	std::transform(after.begin(), after.end(), after_transformed.begin(), [&after_centre](const Common::Point_f& el) {return el - after_centre; });

	Eigen::Matrix<float, 3, Eigen::Dynamic> A(3, before_transformed.size());
	Eigen::Matrix<float, 3, Eigen::Dynamic> B(3, after_transformed.size());
	Eigen::Matrix<float, 3, Eigen::Dynamic> beforeMatrix(3, before.size());

	for (long i = 0; i < before_transformed.size(); i++)
	{
		A(0, i) = before_transformed[i].x;
		A(1, i) = before_transformed[i].y;
		A(2, i) = before_transformed[i].z;
		B(0, i) = after_transformed[i].x;
		B(1, i) = after_transformed[i].y;
		B(2, i) = after_transformed[i].z;
		beforeMatrix(0, i) = before[i].x;
		beforeMatrix(1, i) = before[i].y;
		beforeMatrix(2, i) = before[i].z;
	}

	Eigen::MatrixXf C = B * A.transpose();

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

	Eigen::Matrix<float, 3, 3> matrixU = svd.matrixU();
	Eigen::Matrix<float, 3, 3> matrixV = svd.matrixV();
	Eigen::Matrix<float, 3, 3> matrixVtransposed = matrixV.transpose();

	Eigen::Matrix<float, 3, 3> determinantMatrix = matrixU * matrixVtransposed;

	Eigen::Matrix<float, 3, 3> RotationMatrix = matrixU * Eigen::DiagonalMatrix<float, 3>(1, 1, determinantMatrix.determinant()) * matrixVtransposed;

	Eigen::Vector3f vectorX(before_centre.x, before_centre.y, before_centre.z);
	Eigen::Vector3f vectorY(after_centre.x, after_centre.y, after_centre.z);

	Eigen::Vector3f vectorD = vectorY - RotationMatrix * vectorX;

	Eigen::Matrix<float, 3, Eigen::Dynamic> afterMatrix = RotationMatrix * beforeMatrix;

	std::vector<Common::Point_f> result = std::vector<Common::Point_f>(before.size());

	for (long i = 0; i < before.size(); i++)
	{
		result[i].x = afterMatrix(0, i) + vectorD(0);
		result[i].y = afterMatrix(1, i) + vectorD(1);
		result[i].z = afterMatrix(2, i) + vectorD(2);
	}

	return std::make_tuple(result, RotationMatrix, vectorD);
}
}