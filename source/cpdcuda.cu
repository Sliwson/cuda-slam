#include "cpdcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"
#include "timer.h"
#include "testutils.h"

using namespace CUDACommon;

namespace
{
	typedef thrust::device_vector<glm::vec3> GpuCloud;

	struct Probabilities
	{
		// The probability matrix, multiplied by the identity vector.
		thrust::device_vector<float> p1;
		// The probability matrix, transposed, multiplied by the identity vector.
		thrust::device_vector<float> pt1;
		// The probability matrix multiplied by the fixed(cloud before) points.
		thrust::device_vector<glm::vec3> px;
		// The total error.
		float error;
	};

	float CalculateSigmaSquared(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter)
	{
		if (cloudBefore.size() > cloudAfter.size())
		{
			const auto functor = Functors::CalculateSigmaSquaredInRow(cloudAfter);
			return thrust::transform_reduce(thrust::device, cloudBefore.begin(), cloudBefore.end(), functor, 0.0f, thrust::plus<float>()) / (float)(3 * cloudBefore.size() * cloudAfter.size());
		}
		else
		{
			const auto functor = Functors::CalculateSigmaSquaredInRow(cloudBefore);
			return thrust::transform_reduce(thrust::device, cloudAfter.begin(), cloudAfter.end(), functor, 0.0f, thrust::plus<float>()) / (float)(3 * cloudBefore.size() * cloudAfter.size());
		}
		return -1.0f;
	}

	//Probabilities ComputePMatrix(
	//	const GpuCloud& cloudBefore,
	//	const GpuCloud& cloudTransformed,
	//	const float& constant,
	//	const float& sigmaSquared,
	//	const bool& doTruncate,
	//	float truncate)
	//{
	//	const float multiplier = -0.5f / sigmaSquared;
	//	thrust::device_vector<float> p = thrust::device_vector<float>(cloudTransformed.size());
	//	thrust::device_vector<float> p1 = thrust::device_vector<float>(cloudTransformed.size());
	//	thrust::device_vector<float> pt1 = thrust::device_vector<float>(cloudBefore.size());
	//	thrust::device_vector<glm::vec3> px = thrust::device_vector<glm::vec3>(cloudTransformed.size());

	//	thrust::counting_iterator<int> idxfirst(0);
	//	thrust::counting_iterator<int> idxlast = idxfirst + cloudTransformed.size();

	//	//maybe use auto instead of this
	//	thrust::zip_iterator<thrust::tuple<thrust::device_vector<glm::vec3>::iterator, thrust::counting_iterator<int>>> cloudTransformed_first = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.begin(), idxfirst));
	//	thrust::zip_iterator<thrust::tuple<thrust::device_vector<glm::vec3>::iterator, thrust::counting_iterator<int>>> cloudTransformed_last = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.end(), idxlast));

	//	float error = 0.0;
	//	if (doTruncate)
	//		truncate = std::log(truncate);

	//	for (size_t x = 0; x < cloudBefore.size(); x++)
	//	{
	//		const auto functorDenominator = Functors::CalculateDenominator(cloudBefore[x], p, multiplier, doTruncate, truncate);
	//		//const float denominator = thrust::transform_reduce(thrust::device, cloudTransformed_first, cloudTransformed_last, functorDenominator, constant, thrust::plus<float>());
	//		const float denominator = 1.0f;

	//		pt1[x] = 1.0f - constant / denominator;

	//		const auto functor = Functors::CalculateP1AndPX(cloudBefore[x], p, p1, px, denominator);
	//		thrust::for_each(thrust::device, idxfirst, idxlast, functor);
	//		error -= std::log(denominator);
	//	}
	//	error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2.0f;

	//	return { p1, pt1, px, error };
	//}

	glm::mat4 CudaCPD(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudAfter,
		int* iterations,
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance,
		Common::ApproximationType fgt)
	{
		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		float scale = 1.0f;
		float sigmaSquared = CalculateSigmaSquared(cloudBefore, cloudAfter);
		std::cout << "SIGMA squared " << sigmaSquared << std::endl;
		float sigmaSquared_init = sigmaSquared;

		/*weight = std::clamp(weight, 0.0f, 1.0f);
		if (weight == 0.0f)
			weight = 1e-6f;
		if (weight == 1.0f)
			weight = 1.0f - 1e-6f;*/

			//const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudAfter.size()) / ((1 - weight) * cloudBefore.size());
			//float ntol = tolerance + 10.0f;
			//float l = 0.0f;
			//Probabilities probabilities;
			//GpuCloud transformedCloud = cloudAfter;
			////EM optimization
			//while (*iterations < maxIterations && ntol > tolerance && sigmaSquared > eps)
			//{
			//	//E-step
			//	if (fgt == FastGaussTransform::FGTType::None)
			//		probabilities = ComputePMatrix(cloudBefore, transformedCloud, constant, sigmaSquared, false, -1.0f);
			//	//else
			//	//	probabilities = ComputePMatrixFast(cloudBefore, transformedCloud, constant, weight, &sigmaSquared, sigmaSquared_init, fgt);

			//	ntol = std::abs((probabilities.error - l) / probabilities.error);
			//	l = probabilities.error;

			//	//std::cout << "P1" << std::endl;
			//	//thrust::copy(probabilities.p1.begin(), probabilities.p1.end(), std::ostream_iterator<float>(std::cout, " "));
			//	//std::cout << "Pt1" << std::endl;
			//	//thrust::copy(probabilities.pt1.begin(), probabilities.pt1.end(), std::ostream_iterator<float>(std::cout, " "));
			//	//std::cout << "PX" << std::endl;
			//	//thrust::copy(probabilities.px.begin(), probabilities.px.end(), std::ostream_iterator<glm::vec3>(std::cout, " "));

			//	//M-step
			//	//MStep(probabilities, cloudBefore, cloudAfter, const_scale, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			//	//transformedCloud = GetTransformedCloud(cloudAfter, rotationMatrix, translationVector, scale);
			//	(*error) = sigmaSquared;
			//	(*iterations)++;
			//}
			////return std::make_pair(scale * rotationMatrix, translationVector);
			//
		return glm::mat4(0.0f);
	}
}

void CPDTest()
{
	const char* objectPath = "data/bunny.obj";
	const int pointCount = 10;
	const float testEps = 1e-6f;
	const float weight = 0.0f;
	const bool const_scale = false;
	const int max_iterations = 50;
	const Common::ApproximationType fgt = Common::ApproximationType::None;

	srand(666);
	int iterations = 0;
	float error = 1.0f;
	Timer timer("Cpu timer");

	timer.StartStage("cloud-loading");
	auto cloud = LoadCloud(objectPath);
	timer.StopStage("cloud-loading");
	printf("Cloud size: %d\n", cloud.size());

	timer.StartStage("processing");
	std::transform(cloud.begin(), cloud.end(), cloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 100.f, point.y * 100.f, point.z * 100.f }; });
	if (pointCount > 0)
		cloud.resize(pointCount);

	int cloudSize = cloud.size();
	printf("Processing %d points\n", cloudSize);

	const auto translation_vector = glm::vec3(15.0f, 0.0f, 0.0f);
	const auto rotation_matrix = Tests::GetRotationMatrix({ 1.0f, 0.4f, -0.3f }, glm::radians(50.0f));

	const auto transform = ConvertToTransformationMatrix(rotation_matrix, translation_vector);
	//const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
	const auto permutation = GetRandomPermutationVector(cloudSize);
	auto permutedCloud = ApplyPermutation(cloud, permutation);
	std::transform(permutedCloud.begin(), permutedCloud.end(), permutedCloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 2.f, point.y * 2.f, point.z * 2.f }; });
	const auto transformedCloud = GetTransformedCloud(cloud, transform);
	const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);
	timer.StopStage("processing");

	const auto hostBefore = CommonToThrustVector(transformedPermutedCloud);
	const auto hostAfter = CommonToThrustVector(cloud);

	GpuCloud deviceCloudBefore = hostBefore;
	GpuCloud deviceCloudAfter = hostAfter;

	timer.StartStage("cpd1");
	const auto icpCalculatedTransform1 = CudaCPD(deviceCloudBefore, deviceCloudAfter, &iterations, &error, testEps, weight, const_scale, max_iterations, testEps, fgt);
	timer.StopStage("cpd1");

	//iterations = 0;
	//error = 1.0f;
	//timer.StartStage("icp2");
	////const auto icpCalculatedTransform2 = CoherentPointDrift::GetRigidCPDTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, weigth, const_scale, max_iterations, testEps, fgt);
	//timer.StopStage("icp2");

	//printf("ICP test (%d iterations) error = %g\n", iterations, error);

	//std::cout << "Transform Matrix" << std::endl;
	//PrintMatrix(transform);
	//std::cout << "Inverted Transform Matrix" << std::endl;
	//PrintMatrix(glm::inverse(transform));

	//std::cout << "CPD1 Matrix" << std::endl;
	//PrintMatrix(icpCalculatedTransform1.first, icpCalculatedTransform1.second);

	//timer.PrintResults();

	//Common::Renderer renderer(
	//	Common::ShaderType::SimpleModel,
	//	cloud, //red
	//	transformedPermutedCloud, //green
	//	GetTransformedCloud(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
	//	//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
	//	std::vector<Point_f>(1)); //green

	//renderer.Show();
}
