#include "cuda.cuh"
#include "functors.cuh"

namespace {
	typedef thrust::device_vector<glm::vec3> Cloud;
	typedef thrust::device_vector<int> IndexIterator;
	typedef thrust::permutation_iterator<Cloud, IndexIterator> Permutation;

	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud(vec.size() * 3);
		for (int i = 0; i < vec.size(); i++)
			hostCloud[i] = (glm::vec3)vec[i];

		return hostCloud;
	}

	std::vector<Point_f> ThrustToCommonVector(const Cloud& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud = vec;
		std::vector<Point_f> outVector(vec.size());
		for (int i = 0; i < hostCloud.size(); i++)
			outVector[i] = { hostCloud[i].x, hostCloud[i].y, hostCloud[i].z };

		return outVector;
	}

	glm::vec3 CalculateCentroid(const Cloud& vec)
	{
		const auto sum = thrust::reduce(thrust::device, vec.begin(), vec.end());
		return sum / static_cast<float>(vec.size());
	}

	void TransformCloud(const Cloud& vec, Cloud& out, const glm::mat4& transform)
	{
		const auto functor = Functors::MatrixTransform(transform);
		thrust::transform(thrust::device, vec.begin(), vec.end(), out.begin(), functor);
	}
	
	IndexIterator GetCorrespondingPoints(const Cloud& before, const Cloud& after)
	{
		thrust::device_vector<int> indices(before.size());
		thrust::sequence(thrust::device, indices.begin(), indices.end());
		return indices;
	}

	float GetMeanSquaredError(const IndexIterator& permutation, const Cloud& before, const Cloud& after)
	{
		//auto permutationIterator = thrust::make_permutation_iterator(before, permutation);
		//auto zip = thrust::make_zip_iterator(thrust::make_tuple(before, after));
		return 0.f;
	}

	glm::mat4 CudaICP(const Cloud& before, const Cloud& after)
	{
		const int maxIterations = 60;
		const float TEST_EPS = 1e-5;

		int iterations = 0;
		glm::mat4 transformationMatrix(1.0f);
		Cloud workingBefore(before.size());
		thrust::copy(thrust::device, before.begin(), before.end(), workingBefore.begin());

		while (iterations < maxIterations)
		{
			auto correspondingPoints = GetCorrespondingPoints(workingBefore, after);

			//// Here we multiply
			//transformationMatrix = LeastSquaresSVD(correspondingPoints.first, correspondingPoints.second, error) * transformationMatrix;

			TransformCloud(before, workingBefore, transformationMatrix);
			float error = GetMeanSquaredError(correspondingPoints, workingBefore, after);
			if (error < TEST_EPS)
				break;

			iterations++;
		}

		return transformationMatrix;
	}
}

void CudaTest()
{
	const auto testCloud = LoadCloud("data/bunny.obj");
	const auto hostCloud = CommonToThrustVector(testCloud);

	Cloud deviceCloudBefore = hostCloud;
	Cloud deviceCloudAfter(deviceCloudBefore.size());
	Cloud calculatedCloud(deviceCloudBefore.size());

	const auto scaleInput = Functors::ScaleTransform(100.f);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudBefore.begin(), scaleInput);

	const auto sampleTransform = glm::translate(glm::rotate(glm::mat4(1.f), glm::radians(20.f), { 0.5f, 0.5f, 0.f }), { 5.f, 5.f, 0.f });
	TransformCloud(deviceCloudBefore, deviceCloudAfter, sampleTransform);

	const auto result = CudaICP(deviceCloudBefore, deviceCloudAfter);
	TransformCloud(deviceCloudBefore, calculatedCloud, result);

	Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			ThrustToCommonVector(deviceCloudBefore), //grey
			ThrustToCommonVector(deviceCloudAfter), //blue
			ThrustToCommonVector(calculatedCloud), //red
			std::vector<Point_f>(1));

	renderer.Show();
}
