#include "tests.h"
#include "timer.h"
#include "basicicp.h"
#include "coherentpointdrift.h"
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
		//std::transform(permutedCloud.begin(), permutedCloud.end(), permutedCloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 2.f, point.y * 2.f, point.z * 2.f }; });
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
			cloud, //red
			transformedPermutedCloud, //green
			GetTransformedCloud(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
			GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue

		renderer.Show();
	}

	void BasicICPTest(const char* objectPath1, const char* objectPath2, const int& pointCount1, const int& pointCount2, const float& testEps)
	{
		srand(RANDOM_SEED);
		int iterations = 0;
		float error = 1.0f;

		auto cloud1 = LoadCloud(objectPath1);
		auto cloud2 = LoadCloud(objectPath2);

		printf("First cloud size: %d, Second cloud size: %d\n", cloud1.size(), cloud2.size());

		size_t min_size = std::min(cloud1.size(), cloud2.size());
		cloud1.resize(min_size);
		cloud2.resize(min_size);		

		std::transform(cloud1.begin(), cloud1.end(), cloud1.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud2.begin(), cloud2.end(), cloud2.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		if (pointCount1 > 0)
			cloud1.resize(pointCount1);
		if (pointCount2 > 0)
			cloud1.resize(pointCount2);

		int cloudSize1 = cloud1.size();
		int cloudSize2 = cloud2.size();
		printf("Processing (%d, %d) points\n", cloudSize1, cloudSize2);

		// transformation
		const float scale1 = 1.0f;
		const auto translation_vector1 = glm::vec3(15.0f, 0.0f, 0.0f);
		const auto rotation_matrix1 = GetRotationMatrix({ 1.0f, 0.4f, -0.3f }, glm::radians(50.0f));

		const auto transform1 = ConvertToTransformationMatrix(scale1 * rotation_matrix1, translation_vector1);

		const auto transform2 = glm::mat4(1.0f);
		//const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));

		// permuting both clouds
		const auto permutation1 = GetRandomPermutationVector(cloudSize1);
		const auto permutation2 = GetRandomPermutationVector(cloudSize2);
		auto permutedCloud1 = ApplyPermutation(cloud1, permutation1);
		auto permutedCloud2 = ApplyPermutation(cloud2, permutation2);

		const auto transformedPermutedCloud1 = GetTransformedCloud(permutedCloud1, transform1);
		const auto transformedPermutedCloud2 = GetTransformedCloud(permutedCloud2, transform2);

		// parameters:
		const float max_distance = 1000.0f;
		const int max_iterations = 100;

		auto icp1start = std::chrono::high_resolution_clock::now();
		const auto icpCalculatedTransform1 = BasicICP::GetBasicICPTransformationMatrix(transformedPermutedCloud1, transformedPermutedCloud2, &iterations, &error, testEps, max_distance, max_iterations);
		auto icp1end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> icp1duration = icp1end - icp1start;

		printf("ICP test (%d iterations) error = %g\n", iterations, error);
		printf("ICP test 1 duration %f\n", icp1duration.count());

		std::cout << "Transform Matrix 1" << std::endl;
		PrintMatrix(transform1);
		std::cout << "Inverted Transform Matrix 1" << std::endl;
		PrintMatrix(glm::inverse(transform1));

		std::cout << "ICP1 Matrix" << std::endl;
		PrintMatrix(icpCalculatedTransform1.first, icpCalculatedTransform1.second);

		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			transformedPermutedCloud1, //red
			transformedPermutedCloud2, //green
			GetTransformedCloud(transformedPermutedCloud1, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
			//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
			std::vector<Point_f>(1)); //green

		renderer.Show();
	}

	void RigidCPDTest(const char* objectPath, const int& pointCount, const float& testEps)
	{
		srand(RANDOM_SEED);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		int iterations = 0;
		float error = 1.0f;

		const char* object_path1 = "data/noise_00_bunny.off";
		const char* object_path2 = "data/noise_25_bunny.off";
		const char* object_path3 = "data/noise_50_bunny.off";
		const char* object_path4 = "data/bunny-decapitated.obj";

		//const auto cloud = GetRandomPointCloud(corner, size, 3000);
		auto cloud = LoadCloud(objectPath);
		auto cloud1 = LoadCloud(object_path1);
		auto cloud2 = LoadCloud(object_path2);
		auto cloud3 = LoadCloud(object_path3);
		auto cloud4 = LoadCloud(object_path4);
		printf("Original cloud size: %d, cloud1: %d, cloud2: %d, cloud3: %d, cloud4: %d\n", cloud.size(), cloud1.size(), cloud2.size(), cloud3.size(), cloud4.size());

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud1.begin(), cloud1.end(), cloud1.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud2.begin(), cloud2.end(), cloud2.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud3.begin(), cloud3.end(), cloud3.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud4.begin(), cloud4.end(), cloud4.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		//if (pointCount > 0)
			//cloud.resize(pointCount);

		//cloud.resize(4000);
		//cloud4.resize(4000);

		int cloudSize = cloud.size();
		printf("Processing %d points\n", cloudSize);

		const auto translation_vector = glm::vec3(15.0f, 0.0f, 0.0f);
		const auto rotation_matrix = GetRotationMatrix({ 1.0f, 0.4f, -0.3f }, glm::radians(50.0f));

		const auto transform = ConvertToTransformationMatrix(rotation_matrix, translation_vector);
		//const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		auto permutedCloud = ApplyPermutation(cloud, permutation);
		std::transform(permutedCloud.begin(), permutedCloud.end(), permutedCloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 2.f, point.y * 2.f, point.z * 2.f }; });
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

		printf("TransfPermCloudSize: %d\n", transformedPermutedCloud.size());
		printf("Cloud4Size: %d\n", cloud4.size());

		//TODO: scale clouds to the same size always so threshold would make sense
		auto icp1start = std::chrono::high_resolution_clock::now();
		const auto icpCalculatedTransform1 = CoherentPointDrift::GetRigidCPDTransformationMatrix(transformedPermutedCloud, cloud3, &iterations, &error, testEps, 0.1f, true, 30);
		auto icp2start = std::chrono::high_resolution_clock::now();
		iterations = 0;
		error = 1.0f;
		//const auto icpCalculatedTransform2 = CoherentPointDrift::GetRigidCPDTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, 0.0f, 0);
		auto icp2end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> icp1duration = icp2start - icp1start;
		std::chrono::duration<double> icp2duration = icp2end - icp2start;

		//const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform, testEps);
		//const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform, testEps);

		//printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
		//printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
		printf("ICP test (%d iterations) error = %g\n", iterations, error);
		printf("ICP test 1 duration %f\n", icp1duration.count());
		printf("ICP test 2 duration %f\n", icp2duration.count());

		std::cout << "Transform Matrix" << std::endl;
		PrintMatrix(transform);
		std::cout << "Inverted Transform Matrix" << std::endl;
		PrintMatrix(glm::inverse(transform));

		std::cout << "ICP1 Matrix" << std::endl;
		PrintMatrix(icpCalculatedTransform1.first, icpCalculatedTransform1.second);

		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			cloud3, //red
			transformedPermutedCloud, //green
			GetTransformedCloud(cloud3, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
			//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
			std::vector<Point_f>(1)); //green

		//Common::Renderer renderer(
		//	Common::ShaderType::SimpleModel,
		//	cloud1, //red
		//	cloud2, //green
		//	cloud3, //yellow
		//	//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
		//	cloud4); //blue

		renderer.Show();
	}

	void RigidCPDTest(const char* objectPath1, const char* objectPath2, const int& pointCount1, const int& pointCount2, const float& testEps)
	{
		srand(RANDOM_SEED);
		int iterations = 0;
		float error = 1.0f;

		auto cloud1 = LoadCloud(objectPath1);
		auto cloud2 = LoadCloud(objectPath2);

		printf("First cloud size: %d, Second cloud size: %d\n", cloud1.size(), cloud2.size());

		std::transform(cloud1.begin(), cloud1.end(), cloud1.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		std::transform(cloud2.begin(), cloud2.end(), cloud2.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		if (pointCount1 > 0)
			cloud1.resize(pointCount1);
		if (pointCount2 > 0)
			cloud1.resize(pointCount2);

		int cloudSize1 = cloud1.size();
		int cloudSize2 = cloud2.size();
		printf("Processing (%d, %d) points\n", cloudSize1, cloudSize2);

		// transformation

		const float scale1 = 1.0f;
		const auto translation_vector1 = glm::vec3(15.0f, 0.0f, 0.0f);
		const auto rotation_matrix1 = GetRotationMatrix({ 1.0f, 0.4f, -0.3f }, glm::radians(50.0f));

		const auto transform1 = ConvertToTransformationMatrix(scale1 * rotation_matrix1, translation_vector1);

		const auto transform2 = glm::mat4(1.0f);
		//const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));

		// permuting both clouds
		const auto permutation1 = GetRandomPermutationVector(cloudSize1);
		const auto permutation2 = GetRandomPermutationVector(cloudSize2);
		auto permutedCloud1 = ApplyPermutation(cloud1, permutation1);
		auto permutedCloud2 = ApplyPermutation(cloud2, permutation2);

		const auto transformedPermutedCloud1 = GetTransformedCloud(permutedCloud1, transform1);
		const auto transformedPermutedCloud2 = GetTransformedCloud(permutedCloud2, transform2);

		// parameters:
		const float weight = 0.7f;
		const bool const_scale = false;
		const int max_iterations = 50;

		auto icp1start = std::chrono::high_resolution_clock::now();
		const auto icpCalculatedTransform1 = CoherentPointDrift::GetRigidCPDTransformationMatrix(transformedPermutedCloud1, transformedPermutedCloud2, &iterations, &error, testEps, weight, const_scale, max_iterations);
		auto icp1end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> icp1duration = icp1end - icp1start;

		printf("ICP test (%d iterations) error = %g\n", iterations, error);
		printf("ICP test 1 duration %f\n", icp1duration.count());

		std::cout << "Transform Matrix 1" << std::endl;
		PrintMatrix(transform1);
		std::cout << "Inverted Transform Matrix 1" << std::endl;
		PrintMatrix(glm::inverse(transform1));

		std::cout << "ICP1 Matrix" << std::endl;
		PrintMatrix(icpCalculatedTransform1.first, icpCalculatedTransform1.second);

		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			transformedPermutedCloud2, //red
			transformedPermutedCloud1, //green
			GetTransformedCloud(transformedPermutedCloud2, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
			//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
			std::vector<Point_f>(1)); //green

		renderer.Show();
	}
}