#include "tests.h"
#include "timer.h"
#include "basicicp.h"
#include "noniterative.h"
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

		Common::Timer timer("Cpu timer");

		timer.StartStage("cloud-loading");
		auto cloud = LoadCloud(objectPath);
		timer.StopStage("cloud-loading");

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		if (pointCount > 0)
			cloud.resize(pointCount);

		int cloudSize = cloud.size();
		printf("Processing %d points\n", cloudSize);
		timer.StartStage("processing");

		const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);
		
		timer.StopStage("processing");

		//TODO: scale clouds to the same size always so threshold would make sense
		iterations = 0;
		error = 1.0f;

		timer.StartStage("icp1");
		const auto icpCalculatedTransform1 = BasicICP::GetBasicICPTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, 25.0f, 5);
		timer.StopStage("icp1");

		timer.StartStage("icp2");
		const auto icpCalculatedTransform2 = BasicICP::GetBasicICPTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, 1000.0f, 100);
		timer.StopStage("icp2");

		const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform, testEps);
		const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform, testEps);

		printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
		printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
		printf("ICP test (%d iterations) error = %g\n", iterations, error);

		std::cout << "Transform Matrix" << std::endl;
		PrintMatrix(transform);

		std::cout << "ICP2 Matrix" << std::endl;
		PrintMatrix(icpCalculatedTransform2.first, icpCalculatedTransform2.second);

		timer.PrintResults();

		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			cloud, //grey
			transformedCloud, //blue
			GetTransformedCloud(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //red
			GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //green

		renderer.Show();
	}

	// Randomly transform first pointCount points from 3d object loaded from objectPath using non iterative slam
	void NonIterativeTest(const char* objectPath, const int& pointCount, const float& testEps)
	{
		srand(RANDOM_SEED);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		float error = 1.0f;

		Common::Timer timer("Cpu timer");

		timer.StartStage("cloud-loading");
		auto cloud = LoadCloud(objectPath);
		timer.StopStage("cloud-loading");

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		if (pointCount > 0)
			cloud.resize(pointCount);

		int cloudSize = cloud.size();
		printf("Processing %d points\n", cloudSize);
		timer.StartStage("processing");

		const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

		timer.StopStage("processing");

		//error = 1.0f;

		timer.StartStage("non-iterative-ordered");
		const auto orderedCalculatedTransform = NonIterative::GetNonIterativeTransformationMatrix(cloud, transformedCloud, &error);
		timer.StopStage("non-iterative-ordered");

		timer.StartStage("non-iterative-permuted");
		const auto permutedCalculatedTransform = NonIterative::GetNonIterativeTransformationMatrix(cloud, transformedPermutedCloud, &error);
		timer.StopStage("non-iterative-permuted");

		std::cout << "Transform Matrix" << std::endl;
		PrintMatrix(transform);

		std::cout << "Result matrix (for ordered test case)" << std::endl;
		PrintMatrix(orderedCalculatedTransform.first, orderedCalculatedTransform.second);

		std::cout << "Result matrix (for permuted test case)" << std::endl;
		PrintMatrix(permutedCalculatedTransform.first, permutedCalculatedTransform.second);

		timer.PrintResults();

		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			cloud, //grey
			transformedCloud, //blue
			GetTransformedCloud(cloud, permutedCalculatedTransform.first, permutedCalculatedTransform.second), //red
			GetTransformedCloud(cloud, orderedCalculatedTransform.first, orderedCalculatedTransform.second)); //green

		renderer.Show();
	}
}