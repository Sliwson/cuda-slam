#include "cuda.cuh"

namespace Helpers {
	struct MatrixTransform : thrust::unary_function<glm::vec3, glm::vec3>
	{
		MatrixTransform(const glm::mat4& transform) : transformMatrix(transform) {}

		__device__ __host__ glm::vec3 operator()(const glm::vec3& vector)
		{
			return glm::vec3(transformMatrix * glm::vec4(vector, 1.f));
		}

	private:
		glm::mat4 transformMatrix = glm::mat4(1.f);
	};

	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud(vec.size() * 3);
		for (int i = 0; i < vec.size(); i++)
			hostCloud[i] = (glm::vec3)vec[i];

		return std::move(hostCloud);
	}
}

void CudaTest()
{
	const auto testCloud = LoadCloud("data/bunny.obj");
	auto hostCloud = Helpers::CommonToThrustVector(testCloud);
	thrust::device_vector<glm::vec3> deviceCloudBefore = hostCloud;
	thrust::device_vector<glm::vec3> deviceCloudAfter(deviceCloudBefore.size());

	const auto sampleTransform = glm::translate(glm::mat4(1.f), { 1.f, 0, 0 });
	auto fun = Helpers::MatrixTransform(sampleTransform);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudAfter.begin(), fun);
}
