#include "tests.h"
#include "basicicp.h"
#include "renderer.h"
#include "shadertype.h"
#include "testutils.h"

using namespace Common;

namespace Tests
{
	// Test if error between cloudAfter and cloudBefore transformed using matrix is less than testEps
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat4& matrix, const float& testEps)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, matrix) <= testEps;
	}

	// Test if error between cloudAfter and cloudBefore transformed using rotationMatrix and translationVector is less than testEps
	bool TestTransformOrdered(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps)
	{
		if (cloudBefore.size() != cloudAfter.size())
			return false;
		return GetMeanSquaredError(cloudBefore, cloudAfter, rotationMatrix, translationVector) <= testEps;
	}

	// Test if error between cloudAfter and cloudBefore permuted using permutation and transformed using matrix is less than testEps
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat4& matrix, const float& testEps)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, matrix, testEps);
	}

	// Test if error between cloudAfter and cloudBefore permuted using permutation and transformed using rotationMatrix and translationVector is less than testEps
	bool TestTransformWithPermutation(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, const std::vector<int>& permutation, const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& testEps)
	{
		const auto permutedCloudBefore = ApplyPermutation(cloudBefore, permutation);
		return TestTransformOrdered(permutedCloudBefore, cloudAfter, rotationMatrix, translationVector, testEps);
	}

	// Randomly transform first pointCount points from 3d object loaded from objectPath using testEps for error evaluation
	void BasicICPTest(const char* objectPath, const int& pointCount, const float& testEps)
	{
		srand(RANDOM_SEED);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		int iterations = 0;
		float error = 1.0f;

		//const auto cloud = GetRandomPointCloud(corner, size, 3000);
		auto cloud = LoadCloud(objectPath);

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		if (pointCount > 0)
			cloud.resize(pointCount);

		int cloudSize = cloud.size();
		printf("Processing %d points\n", cloudSize);

		const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

		//TODO: scale clouds to the same size always so threshold would make sense
		auto icp1start = std::chrono::high_resolution_clock::now();
		const auto icpCalculatedTransform1 = BasicICP::GetBasicICPTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, 25.0f, 5);
		auto icp2start = std::chrono::high_resolution_clock::now();
		iterations = 0;
		error = 1.0f;
		const auto icpCalculatedTransform2 = BasicICP::GetBasicICPTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, 1000.0f, 100);
		auto icp2end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> icp1duration = icp2start - icp1start;
		std::chrono::duration<double> icp2duration = icp2end - icp2start;

		const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform, testEps);
		const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform, testEps);

		printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
		printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
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