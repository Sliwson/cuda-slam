#include "cuda.cuh"
#include "functors.cuh"

namespace {
	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud(vec.size() * 3);
		for (int i = 0; i < vec.size(); i++)
			hostCloud[i] = (glm::vec3)vec[i];

		return hostCloud;
	}

	std::vector<Point_f> ThrustToCommonVector(const thrust::device_vector<glm::vec3>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud = vec;
		std::vector<Point_f> outVector(vec.size());
		for (int i = 0; i < hostCloud.size(); i++)
			outVector[i] = { hostCloud[i].x, hostCloud[i].y, hostCloud[i].z };

		return outVector;
	}

	glm::vec3 CalculateCentroid(const thrust::device_vector<glm::vec3>& vec)
	{
		const auto sum = thrust::reduce(thrust::device, vec.begin(), vec.end());
		return sum / static_cast<float>(vec.size());
	}

	glm::mat4 CudaICP(const thrust::device_vector<glm::vec3>& before, const thrust::device_vector<glm::vec3>& after)
	{
		return glm::mat4(1.f);
	}
}

void CudaTest()
{
	const auto testCloud = LoadCloud("data/bunny.obj");
	const auto hostCloud = CommonToThrustVector(testCloud);

	thrust::device_vector<glm::vec3> deviceCloudBefore = hostCloud;
	thrust::device_vector<glm::vec3> deviceCloudAfter(deviceCloudBefore.size());
	thrust::device_vector<glm::vec3> calculatedCloud(deviceCloudBefore.size());

	const auto scaleInput = Functors::ScaleTransform(100.f);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudBefore.begin(), scaleInput);

	const auto sampleTransform = glm::translate(glm::rotate(glm::mat4(1.f), glm::radians(20.f), { 0.5f, 0.5f, 0.f }), { 5.f, 5.f, 0.f });
	const auto idealTransform = Functors::MatrixTransform(sampleTransform);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudAfter.begin(), idealTransform);

	const auto result = CudaICP(deviceCloudBefore, deviceCloudAfter);
	const auto calculatedTransform = Functors::MatrixTransform(result);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), calculatedCloud.begin(), calculatedTransform);

	Common::Renderer renderer(
			Common::ShaderType::SimpleModel,
			ThrustToCommonVector(deviceCloudBefore), //grey
			ThrustToCommonVector(deviceCloudAfter), //blue
			ThrustToCommonVector(calculatedCloud), //red
			std::vector<Point_f>(1));

	renderer.Show();
}
