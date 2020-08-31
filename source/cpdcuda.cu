#include "cpdcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"
#include "timer.h"
#include "testutils.h"
#include "cudaprobabilities.h"
#include "mstepparams.cuh"
#include "common.h"
#include "cpdutils.h"

using namespace CUDACommon;
using namespace MStepParams;

namespace
{
	typedef thrust::device_vector<glm::vec3> GpuCloud;

	float CalculateSigmaSquared(const GpuCloud& cloudBefore, const GpuCloud& cloudAfter);
	void ComputePMatrix(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudTransformed,
		CUDAProbabilities::Probabilities& probabilities,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate);
	void ComputePMatrixFast(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudTransformed,
		CUDAProbabilities::Probabilities& probabilities,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const ApproximationType& fgt,
		const std::vector<Point_f>& cloudBeforeCPU,
		const std::vector<Point_f>& cloudAfterCPU,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale);
	void ComputePMatrixWithFGTOnCPU(
		const std::vector<Point_f>& cloudBeforeCPU,
		const std::vector<Point_f>& cloudAfterCPU,
		CUDAProbabilities::Probabilities& probabilities,
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale);
	void MStep(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudAfter,
		const CUDAProbabilities::Probabilities& probabilities,
		CUDAMStepParams& params,
		const bool& const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

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

	void ComputePMatrix(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudTransformed,
		CUDAProbabilities::Probabilities& probabilities,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate)
	{
		const float multiplier = -0.5f / sigmaSquared;

		thrust::fill(thrust::device, probabilities.p1.begin(), probabilities.p1.end(), 0.0f);
		thrust::fill(thrust::device, probabilities.px.begin(), probabilities.px.end(), glm::vec3(0.0f));

		thrust::counting_iterator<int> idxfirst = thrust::make_counting_iterator<int>(0);
		thrust::counting_iterator<int> idxlast = thrust::make_counting_iterator<int>(cloudTransformed.size());

		auto cloudTransformed_first = thrust::make_zip_iterator<>(thrust::make_tuple(cloudTransformed.begin(), idxfirst));
		auto cloudTransformed_last = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.end(), idxlast));

		probabilities.error = 0.0f;
		if (doTruncate)
			truncate = std::log(truncate);

		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			const auto functorDenominator = Functors::CalculateDenominator(cloudBefore[x], probabilities.p, multiplier, doTruncate, truncate);
			thrust::transform(thrust::device, cloudTransformed_first, cloudTransformed_last, probabilities.tmp.begin(), functorDenominator);
			const float denominator = thrust::reduce(thrust::device, probabilities.tmp.begin(), probabilities.tmp.end(), constant, thrust::plus<float>());
			probabilities.pt1[x] = 1.0f - constant / denominator;

			const auto functor = Functors::CalculateP1AndPX(cloudBefore[x], probabilities.p, probabilities.p1, probabilities.px, denominator);
			thrust::for_each(thrust::device, idxfirst, idxlast, functor);
			probabilities.error -= std::log(denominator);
		}
		probabilities.error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2.0f;
	}

	void ComputePMatrixFast(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudTransformed,
		CUDAProbabilities::Probabilities& probabilities,
		const float& constant,
		const float& weight,
		float* sigmaSquared,
		const float& sigmaSquaredInit,
		const ApproximationType& fgt,
		const std::vector<Point_f>& cloudBeforeCPU,
		const std::vector<Point_f>& cloudAfterCPU,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale)
	{
		if (fgt == ApproximationType::Full)
		{
			if (*sigmaSquared < 0.05)
				*sigmaSquared = 0.05;
			ComputePMatrixWithFGTOnCPU(cloudBeforeCPU, cloudAfterCPU, probabilities, weight, *sigmaSquared, sigmaSquaredInit, rotationMatrix, translationVector, scale);
		}
		else if (fgt == ApproximationType::Hybrid)
		{
			if (*sigmaSquared > 0.015 * sigmaSquaredInit)
				ComputePMatrixWithFGTOnCPU(cloudBeforeCPU, cloudAfterCPU, probabilities, weight, *sigmaSquared, sigmaSquaredInit, rotationMatrix, translationVector, scale);
			else
				ComputePMatrix(cloudBefore, cloudTransformed, probabilities, constant, *sigmaSquared, true, 1e-3f);
		}
	}

	void ComputePMatrixWithFGTOnCPU(
		const std::vector<Point_f>& cloudBeforeCPU,
		const std::vector<Point_f>& cloudAfterCPU,
		CUDAProbabilities::Probabilities& probabilities,
		const float& weight,
		const float& sigmaSquared,
		const float& sigmaSquaredInit,
		const glm::mat3& rotationMatrix,
		const glm::vec3& translationVector,
		const float& scale)
	{
		auto cloudTransformedCPU = Common::GetTransformedCloud(cloudAfterCPU, rotationMatrix, translationVector, scale);
		auto prob = CPDutils::ComputePMatrixWithFGT(cloudBeforeCPU, cloudTransformedCPU, weight, sigmaSquared, sigmaSquaredInit);
		Eigen::Matrix<float, -1, 3, Eigen::RowMajor> px = prob.px;
		cudaMemcpy(thrust::raw_pointer_cast(probabilities.p1.data()), prob.p1.data(), cloudAfterCPU.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(probabilities.pt1.data()), prob.pt1.data(), cloudBeforeCPU.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(probabilities.px.data()), px.data(), cloudAfterCPU.size() * 3 * sizeof(float), cudaMemcpyHostToDevice);
		probabilities.error = prob.error;
	}

	void MStep(
		const GpuCloud& cloudBefore,
		const GpuCloud& cloudAfter,
		const CUDAProbabilities::Probabilities& probabilities,
		CUDAMStepParams& params,
		const bool& const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared)
	{
		const float alpha = 1.f, beta = 0.f;
		const int beforeSize = cloudBefore.size();
		const int afterSize = cloudAfter.size();
		const float Np = thrust::reduce(thrust::device, probabilities.p1.begin(), probabilities.p1.end(), 0.0f, thrust::plus<float>());
		const float InvertedNp = 1.0f / Np;

		//create array beforeT
		auto countingBeforeBegin = thrust::make_counting_iterator<int>(0);
		auto countingBeforeEnd = thrust::make_counting_iterator<int>(beforeSize);
		auto zipBeforeBegin = thrust::make_zip_iterator(thrust::make_tuple(countingBeforeBegin, cloudBefore.begin()));
		auto zipBeforeEnd = thrust::make_zip_iterator(thrust::make_tuple(countingBeforeEnd, cloudBefore.end()));

		auto convertBefore = Functors::GlmToCuBlas(false, beforeSize, params.beforeT);
		thrust::for_each(thrust::device, zipBeforeBegin, zipBeforeEnd, convertBefore);

		//create array afterT
		auto countingAfterBegin = thrust::make_counting_iterator<int>(0);
		auto countingAfterEnd = thrust::make_counting_iterator<int>(afterSize);
		auto zipAfterBegin = thrust::make_zip_iterator(thrust::make_tuple(countingAfterBegin, cloudAfter.begin()));
		auto zipAfterEnd = thrust::make_zip_iterator(thrust::make_tuple(countingAfterEnd, cloudAfter.end()));

		auto convertAfter = Functors::GlmToCuBlas(false, afterSize, params.afterT);
		thrust::for_each(thrust::device, zipAfterBegin, zipAfterEnd, convertAfter);

		//create array px
		auto countingPXBegin = thrust::make_counting_iterator<int>(0);
		auto countingPXEnd = thrust::make_counting_iterator<int>(probabilities.px.size());
		auto zipPXBegin = thrust::make_zip_iterator(thrust::make_tuple(countingPXBegin, probabilities.px.begin()));
		auto zipPXEnd = thrust::make_zip_iterator(thrust::make_tuple(countingPXEnd, probabilities.px.end()));

		auto convertPX = Functors::GlmToCuBlas(true, probabilities.px.size(), params.px);
		thrust::for_each(thrust::device, zipPXBegin, zipPXEnd, convertPX);

		cublasSgemv(params.multiplyHandle, CUBLAS_OP_N, 3, beforeSize, &InvertedNp, params.beforeT, 3, params.pt1, 1, &beta, params.centerBefore, 1);

		cublasSgemv(params.multiplyHandle, CUBLAS_OP_N, 3, afterSize, &InvertedNp, params.afterT, 3, params.p1, 1, &beta, params.centerAfter, 1);

		cublasSgemm(params.multiplyHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, afterSize, &alpha, params.afterT, 3, params.px, afterSize, &beta, params.afterTxPX, 3);

		cublasSgemm(params.multiplyHandle, CUBLAS_OP_N, CUBLAS_OP_T, 3, 3, 1, &Np, params.centerBefore, 3, params.centerAfter, 3, &beta, params.centerBeforexCenterAfter, 3);

		float minus = -1.0f;
		cublasSgeam(params.multiplyHandle, CUBLAS_OP_T, CUBLAS_OP_N, 3, 3, &alpha, params.afterTxPX, 3, &minus, params.centerBeforexCenterAfter, 3, params.AMatrix, 3);

		//TODO: try jacobi svd
		//SVD
		cusolverDnSgesvd(params.solverHandle, 'A', 'A', 3, 3, params.AMatrix, 3, params.S, params.U, 3, params.VT, 3, params.work, params.workSize, nullptr, params.devInfo);
		int svdResultInfo = 0;
		cudaMemcpy(&svdResultInfo, params.devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		if (svdResultInfo != 0)
			printf("Svd execution failed!\n");

		float hostS[3], hostVT[9], hostU[9];
		const int copySize = 9 * sizeof(float);
		cudaMemcpy(hostS, params.S, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostVT, params.VT, copySize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostU, params.U, copySize, cudaMemcpyDeviceToHost);

		auto gVT = glm::transpose(CreateGlmMatrix(hostVT));
		auto gU = glm::transpose(CreateGlmMatrix(hostU));

		//revert signs to match svd cpu solution
		for (int i = 0; i < 3; i++)
		{
			gU[0][i] = -gU[0][i];
			gU[1][i] = -gU[1][i];
			gVT[i][0] = -gVT[i][0];
			gVT[i][1] = -gVT[i][1];
		}

		const float determinant = glm::determinant(gU * gVT);
		const auto diagonal = glm::diagonal3x3(glm::vec3{ 1.f, 1.f, determinant });
		*rotationMatrix = gU * diagonal * gVT;
		const auto scaleNumeratorMatrix = glm::diagonal3x3(glm::make_vec3(hostS)) * diagonal;
		const float scaleNumerator = scaleNumeratorMatrix[0][0] + scaleNumeratorMatrix[1][1] + scaleNumeratorMatrix[2][2];

		auto countingSigmaSubtrahendBegin = thrust::make_counting_iterator<int>(0);
		auto countingSigmaSubtrahendEnd = thrust::make_counting_iterator<int>(3 * beforeSize);
		auto zipSigmaSubtrahendBegin = thrust::make_zip_iterator(thrust::make_tuple(countingSigmaSubtrahendBegin, params.beforeT));
		auto zipSigmaSubtrahendEnd = thrust::make_zip_iterator(thrust::make_tuple(countingSigmaSubtrahendEnd, params.beforeT + 3 * beforeSize));

		auto calculateSigmaSubtrahend = Functors::CalculateSigmaSubtrahend(params.pt1);

		thrust::transform(thrust::device, zipSigmaSubtrahendBegin, zipSigmaSubtrahendEnd, params.beforeT, calculateSigmaSubtrahend);
		float sigmaSubtrahend = thrust::reduce(thrust::device, params.beforeT, params.beforeT + 3 * beforeSize, 0.0f, thrust::plus<float>());

		auto countingScaleDenominatorBegin = thrust::make_counting_iterator<int>(0);
		auto countingScaleDenominatorEnd = thrust::make_counting_iterator<int>(3 * afterSize);
		auto zipScaleDenominatorBegin = thrust::make_zip_iterator(thrust::make_tuple(countingScaleDenominatorBegin, params.afterT));
		auto zipScaleDenominatorEnd = thrust::make_zip_iterator(thrust::make_tuple(countingScaleDenominatorEnd, params.afterT + 3 * afterSize));

		auto calculateScaleDenominator = Functors::CalculateSigmaSubtrahend(params.p1);

		thrust::transform(thrust::device, zipScaleDenominatorBegin, zipScaleDenominatorEnd, params.afterT, calculateScaleDenominator);
		float scaleDenominator = thrust::reduce(thrust::device, params.afterT, params.afterT + 3 * afterSize, 0.0f, thrust::plus<float>());

		float hostCenterBefore[3], hostCenterAfter[3];
		cudaMemcpy(hostCenterBefore, params.centerBefore, 3 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostCenterAfter, params.centerAfter, 3 * sizeof(float), cudaMemcpyDeviceToHost);

		auto glmCenterBefore = glm::make_vec3(hostCenterBefore);
		auto glmCenterAfter = glm::make_vec3(hostCenterAfter);

		sigmaSubtrahend -= Np * glm::dot(glmCenterBefore, glmCenterBefore);
		scaleDenominator -= Np * glm::dot(glmCenterAfter, glmCenterAfter);

		if (const_scale == false)
		{
			*scale = scaleNumerator / scaleDenominator;
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / (float)DIMENSION;
		}
		else
		{
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend + scaleDenominator - 2 * scaleNumerator)) / (float)DIMENSION;
		}

		*translationVector = glmCenterBefore - (*scale) * (*rotationMatrix) * glmCenterAfter;
	}

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
		Common::ApproximationType fgt,
		const std::vector<Point_f>& cloudBeforeCPU,
		const std::vector<Point_f>& cloudAfterCPU)
	{
		//allocate memory
		CUDAProbabilities::Probabilities probabilities(cloudBefore.size(), cloudAfter.size());
		CUDAMStepParams mStepParams(cloudBefore.size(), cloudAfter.size(), probabilities);

		*iterations = 0;
		*error = 1e5;
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		float scale = 1.0f;
		float sigmaSquared = CalculateSigmaSquared(cloudBefore, cloudAfter);
		float sigmaSquared_init = sigmaSquared;

		if (weight <= 0.0f)
			weight = 1e-6f;
		if (weight >= 1.0f)
			weight = 1.0f - 1e-6f;

		const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudAfter.size()) / ((1 - weight) * cloudBefore.size());
		float ntol = tolerance + 10.0f;
		float l = 0.0f;
		GpuCloud transformedCloud = cloudAfter;
		//EM optimization
		while (*iterations < maxIterations && ntol > tolerance && sigmaSquared > eps)
		{
			//E-step
			if (fgt == Common::ApproximationType::None)
				ComputePMatrix(cloudBefore, transformedCloud, probabilities, constant, sigmaSquared, false, -1.0f);
			else
				ComputePMatrixFast(cloudBefore, transformedCloud, probabilities, constant, weight, &sigmaSquared, sigmaSquared_init, fgt, cloudBeforeCPU, cloudAfterCPU, rotationMatrix, translationVector, scale);

			ntol = std::abs((probabilities.error - l) / probabilities.error);
			l = probabilities.error;

			//M-step
			MStep(cloudBefore, cloudAfter, probabilities, mStepParams, const_scale, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			TransformCloud(cloudAfter, transformedCloud, ConvertToTransformationMatrix(scale * rotationMatrix, translationVector));
			(*error) = sigmaSquared;
			(*iterations)++;
		}
		mStepParams.Free();
		return ConvertToTransformationMatrix(scale * rotationMatrix, translationVector);
	}
}

void CPDTest()
{
	srand(666);
	const char* objectPath = "data/bunny.obj";
	int pointCount = 5000;
	float testEps = 1e-4f;
	float weight = 0.1f;
	bool const_scale = false;
	const int max_iterations = 50;
	Common::ApproximationType fgt = Common::ApproximationType::Hybrid;

	//reading data from terminal
	//
	/*
	int weight_int = 0;
	int const_scale_int = 0;
	int fgt_int = 0;

	std::cout << "Point count" << std::endl;
	std::cin >> pointCount;
	std::cout << "Weight * 100" << std::endl;
	std::cin >> weight_int;
	std::cout << "const_scale [0-false, 1-true]" << std::endl;
	std::cin >> const_scale_int;
	std::cout << "fgt type [0-None, 1-Hybrid, 2-Full]" << std::endl;
	std::cin >> fgt_int;

	weight = (float)weight_int / 100.0f;

	if (const_scale_int == 0) const_scale = false;
	if (const_scale_int == 1) const_scale = true;

	if (fgt_int == 0) fgt = Common::ApproximationType::None;
	if (fgt_int == 1) fgt = Common::ApproximationType::Hybrid;
	if (fgt_int == 2) fgt = Common::ApproximationType::Full;

	std::cout << "Point count: " << pointCount << " weight: " << weight << " const_scale: " << const_scale << " fgt: " << (int)(fgt) << std::endl;
	//
	*/
	const float scale = 2.0f;

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

	const auto transform = ConvertToTransformationMatrix(scale * rotation_matrix, translation_vector);
	//const auto transform = GetRandomTransformMatrix({ 0.f, 0.f, 0.f }, { 10.0f, 10.0f, 10.0f }, glm::radians(35.f));
	const auto permutation = GetRandomPermutationVector(cloudSize);
	auto permutedCloud = ApplyPermutation(cloud, permutation);
	//std::transform(permutedCloud.begin(), permutedCloud.end(), permutedCloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 2.f, point.y * 2.f, point.z * 2.f }; });
	const auto transformedCloud = GetTransformedCloud(cloud, transform);
	const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);
	timer.StopStage("processing");

	const auto hostBefore = CommonToThrustVector(transformedPermutedCloud);
	const auto hostAfter = CommonToThrustVector(cloud);

	GpuCloud deviceCloudBefore = hostBefore;
	GpuCloud deviceCloudAfter = hostAfter;

	timer.StartStage("cpd1");
	const auto icpCalculatedTransform1 = CudaCPD(deviceCloudBefore, deviceCloudAfter, &iterations, &error, testEps, weight, const_scale, max_iterations, testEps, fgt, transformedPermutedCloud, cloud);
	timer.StopStage("cpd1");

	printf("ICP test (%d iterations) error = %g\n", iterations, error);

	std::cout << "Transform Matrix" << std::endl;
	PrintMatrix(transform);
	std::cout << "Inverted Transform Matrix" << std::endl;
	PrintMatrix(glm::inverse(transform));

	std::cout << "CPD1 Matrix" << std::endl;
	PrintMatrix(icpCalculatedTransform1);

	timer.PrintResults();

	//std::cout << "Before" << std::endl;
	//PrintVector(deviceCloudBefore);
	//std::cout << "After" << std::endl;
	//PrintVector(deviceCloudAfter);

	Common::Renderer renderer(
		Common::ShaderType::SimpleModel,
		cloud, //red
		transformedPermutedCloud, //green
		GetTransformedCloud(cloud, icpCalculatedTransform1), //yellow
		//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
		std::vector<Point_f>(1)); //green

	renderer.Show();
}
