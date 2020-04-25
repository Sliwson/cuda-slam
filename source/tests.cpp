#include "tests.h"
#include "basicicp.h"
#include "renderer.h"
#include "shadertype.h"

using namespace Common;

namespace Tests
{
	namespace
	{
		constexpr int RANDOM_SEED = 666;
		constexpr float TEST_EPS = 1e-4f;
		const char* object_path = "data/bunny.obj";

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
	}

	//helpers
	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count);
	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRange);
	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationAngle);
	glm::mat3 GetRandomRotationMatrix(float rotationRange);
	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngle);
	std::vector<int> GetRandomPermutationVector(int size);
	std::vector<int> InversePermutation(const std::vector<int>& permutation);
	std::vector<Point_f> ApplyPermutation(const std::vector<Point_f>& input, const std::vector<int>& permutation);
	std::vector<int> GetClosestPointIndexes(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter);
	void PrintMatrix(glm::mat4 matrix);
	void PrintMatrix(glm::mat3 matrix, glm::vec3 vector);
	//helper test functions
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix);
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix);
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector);
	bool TestPermutation(const std::vector<int>& expected, const std::vector<int>& actual);

	std::vector<Point_f> GetRandomPointCloud(const Point_f& corner, const Point_f& size, int count)
	{
		std::vector<Point_f> result;
		for (int i = 0; i < count; i++)
			result.push_back(GetRandomPoint(corner, corner + size));
		return result;
	}

	glm::mat4 GetRandomTransformMatrix(const Point_f& translationMin, const Point_f& translationMax, float rotationRange)
	{
		const auto rotation = glm::mat4(GetRandomRotationMatrix(rotationRange));
		return glm::translate(rotation, glm::vec3(GetRandomPoint(translationMin, translationMax)));
	}

	glm::mat4 GetTranformMatrix(const Point_f& translation, const Point_f& rotationAxis, float rotationAngle)
	{
		const auto rotation = glm::mat4(GetRotationMatrix(rotationAxis, rotationAngle));
		return glm::translate(rotation, glm::vec3(translation));
	}

	glm::mat3 GetRandomRotationMatrix(float rotationRange)
	{
		const auto angle = GetRandomFloat(0, rotationRange);
		//const auto angle = glm::radians(30.0f);
		const auto rotation = glm::rotate(glm::mat4(1.0f), angle, glm::vec3(GetRandomPoint(Point_f::Zero(), Point_f::One())));
		return glm::mat3(rotation);
	}

	glm::mat3 GetRotationMatrix(const Point_f& rotationAxis, float rotationAngle)
	{
		const auto rotation = glm::rotate(glm::mat4(1.0f), rotationAngle, glm::normalize(glm::vec3(rotationAxis)));
		return glm::mat3(rotation);
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

	std::vector<int> GetClosestPointIndexes(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		const auto size = cloudBefore.size();
		std::vector<int> resultPermutation(size);
		return resultPermutation;
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

	void PrintMatrix(glm::mat3 matrix, glm::vec3 vector)
	{
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				std::cout << matrix[j][i] << '\t';
			}
			std::cout << vector[i];
			std::cout << std::endl;
		}
		std::cout << "0\t0\t0\t1\t" << std::endl;
	}

	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, matrix) <= TEST_EPS;
	}

	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, rotationMatrix, translationVector) <= TEST_EPS;
	}

	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, matrix);
	}

	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, rotationMatrix, translationVector);
	}

	bool TestPermutation(const std::vector<int>& expected, const std::vector<int>& actual)
	{
		return actual == expected;
	}

	void BasicICPTest()
	{
		srand(RANDOM_SEED);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		int iterations = 0;
		float error = 1.0f;

		//const auto cloud = GetRandomPointCloud(corner, size, 3000);
		auto cloud = LoadCloud(object_path);

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		cloud.resize(3000);
		int cloudSize = cloud.size();

		const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

		const auto calculatedPermutation = InversePermutation(GetClosestPointIndexes(cloud, transformedPermutedCloud));
		//TODO: scale clouds to the same size always so threshold would make sense
		auto icp1start = std::chrono::high_resolution_clock::now();
		const auto icpCalculatedTransform1 = BasicICP::BasicICP(cloud, transformedPermutedCloud, &iterations, &error, TEST_EPS, 25.0f, 5);
		auto icp2start = std::chrono::high_resolution_clock::now();
		iterations = 0;
		error = 1.0f;
		const auto icpCalculatedTransform2 = BasicICP::BasicICP(cloud, transformedPermutedCloud, &iterations, &error, TEST_EPS, 1000.0f, 100);
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
			GetTransformedCloud(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //red
			GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //green

		renderer.Show();
	}
}