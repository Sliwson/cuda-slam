#include <Eigen/Dense>

#include "common.h"
#include "loader.h"

namespace Common 
{
	namespace
	{
		constexpr float TEST_EPS = 1e-6f;

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
			GetRandomFloat(0, rotationRange),
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

	Point_f GetCenterOfMass(const std::vector<Point_f> & cloud)
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
			diffSum += (diff.Length() * diff.Length());
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

	std::pair<std::vector<Point_f>, std::vector<Point_f>>GetClosestPointPair(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{

		return std::make_pair(cloudBefore, cloudAfter);
	}

	std::vector<Point_f> GetAlignedCloud(const std::vector<Point_f>& cloud)
	{
		auto center = GetCenterOfMass(cloud);
		auto result = std::vector<Point_f>(cloud.size());
		// TODO: Use thrust maybe and parametrize host/device run


		return result;
	}

	Eigen::Matrix3Xf GetMatrix3XFromPointsVector(const std::vector<Point_f>& points)
	{
		Eigen::Matrix3Xf result = Eigen::ArrayXXf::Zero(3, points.size());


		return result;
	}

	glm::mat4 LeastSquaresSVD(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& orderedCloudAfter, const Point_f& cBefore, const Point_f& cAfter)
	{
		auto transformationMatrix = glm::mat4();
		
		// Official documentation says thin U and thin V are enough for us, not gonna argue
		// But maybe it is not enough, delete flags then
		//Eigen::JacobiSVD svd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);

		//Eigen::Matrix3f vMatrix = svd.matrixV();
		//Eigen::Matrix3f uMatrix = svd.matrixU();

		return transformationMatrix;
	}

	glm::mat4 GetTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int *iterations, float *error, int maxIterations = -1)
	{
		glm::mat4 transformationMatrix(1.0f);
		std::pair<std::vector<Point_f>, std::vector<Point_f>> closestPoints;
		*iterations = 0;
		*error = 1.0f;

		return transformationMatrix;
	}

	void LibraryTest()
	{
		srand(666);
		const Point_f corner = { -1, -1, -1 };
		const Point_f size = { 2, 2, 2 };
		int iterations = 0;
		float error = 1.0f;

		//const auto cloud = GetRandomPointCloud(corner, size, 3000);
		auto cloud = LoadCloud("data/bunny.obj");

		std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f { point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
		cloud.resize(3000);
		int cloudSize = cloud.size();

		const auto transform = glm::translate(glm::mat4(1.f), { 1.f, 1.f, 0.f }); //GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 0.1, 0.1, 0.1 }, glm::radians(0.f));
		const auto permutation = GetRandomPermutationVector(cloudSize);
		const auto permutedCloud = ApplyPermutation(cloud, permutation);
		const auto transformedCloud = GetTransformedCloud(cloud, transform);
		const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);

		const auto calculatedPermutation = InversePermutation(GetClosestPointIndexes(cloud, transformedPermutedCloud));
		const auto icpCalculatedTransform1 = GetTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, 50);
		const auto icpCalculatedTransform2 = GetTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, 5);

		const auto resultOrdered = TestTransformOrdered(cloud, transformedCloud, transform);
		const auto resultUnordered = TestTransformWithPermutation(cloud, transformedPermutedCloud, permutation, transform);
		const auto resultPermutation = TestPermutation(permutation, calculatedPermutation);

		printf("Ordered cloud test [%s]\n", resultOrdered ? "OK" : "FAIL");
		printf("Unordered cloud test [%s]\n", resultUnordered ? "OK" : "FAIL");
		printf("Permutation find test [%s]\n", resultPermutation ? "OK" : "FAIL");
		printf("ICP test (%d iterations) error = %g\n", iterations, error);


		Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			cloud, //grey
			transformedCloud, //blue
			GetTransformedCloud(cloud, icpCalculatedTransform1), //red
			GetTransformedCloud(cloud, icpCalculatedTransform2)); //green

		renderer.Show();
	}
}
