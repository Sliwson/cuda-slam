#include "cpdcuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"
#include "timer.h"
#include "testutils.h"
#include "cudaprobabilities.h"
#include "mstepparams.cuh"
#include "common.h"

using namespace CUDACommon;
using namespace CUDAProbabilities;
using namespace MStepParams;

namespace
{
	typedef thrust::device_vector<glm::vec3> Cloud;

	float CalculateSigmaSquared(const Cloud& cloudBefore, const Cloud& cloudAfter);
	void ComputePMatrix(
		const Cloud& cloudBefore,
		const Cloud& cloudTransformed,
		Probabilities& probabilities,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate);
	void MStep(
		const Cloud& cloudBefore,
		const Cloud& cloudAfter,
		const Probabilities& probabilities,
		CUDAMStepParams& params,
		const bool& const_scale,
		glm::mat3* rotationMatrix,
		glm::vec3* translationVector,
		float* scale,
		float* sigmaSquared);

	float CalculateSigmaSquared(const Cloud& cloudBefore, const Cloud& cloudAfter)
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
		const Cloud& cloudBefore,
		const Cloud& cloudTransformed,
		Probabilities& probabilities,
		const float& constant,
		const float& sigmaSquared,
		const bool& doTruncate,
		float truncate)
	{
		const float multiplier = -0.5f / sigmaSquared;
		/*thrust::device_vector<float> p(cloudTransformed.size());
		thrust::device_vector<float> p1(cloudTransformed.size());
		thrust::device_vector<float> pt1(cloudBefore.size());
		thrust::device_vector<glm::vec3> px(cloudTransformed.size());
		thrust::device_vector<float> tmp(cloudTransformed.size());*/

		thrust::counting_iterator<int> idxfirst = thrust::make_counting_iterator<int>(0);
		thrust::counting_iterator<int> idxlast = thrust::make_counting_iterator<int>(cloudTransformed.size());

		////maybe use auto instead of this
		//thrust::zip_iterator<thrust::tuple<Cloud::iterator, thrust::counting_iterator<int>>> cloudTransformed_first = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.begin(), idxfirst));
		auto cloudTransformed_first = thrust::make_zip_iterator<>(thrust::make_tuple(cloudTransformed.begin(), idxfirst));
		auto cloudTransformed_last = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.end(), idxlast));
		//thrust::zip_iterator<thrust::tuple<Cloud::iterator, thrust::counting_iterator<int>>> cloudTransformed_last = thrust::make_zip_iterator(thrust::make_tuple(cloudTransformed.end(), idxlast));

		//auto cloudTransformed_first = thrust::make_zip_iterator(thrust::make_tuple(p.begin(), idxfirst));
		//auto cloudTransformed_last = thrust::make_zip_iterator(thrust::make_tuple(p.end(), idxlast));

		probabilities.error = 0.0f;
		if (doTruncate)
			truncate = std::log(truncate);

		for (size_t x = 0; x < cloudBefore.size(); x++)
		{
			const auto functorDenominator = Functors::CalculateDenominator(cloudBefore[x], probabilities.p, multiplier, doTruncate, truncate);
			//const auto functorDenominator = Functors::CalculateDenominator2();
			//const float denominator = thrust::transform_reduce(thrust::device, cloudTransformed_first, cloudTransformed_last, functorDenominator, constant, thrust::plus<float>());
			thrust::transform(thrust::device, cloudTransformed_first, cloudTransformed_last, probabilities.tmp.begin(), functorDenominator);
			const float denominator = thrust::reduce(thrust::device, probabilities.tmp.begin(), probabilities.tmp.end(), constant, thrust::plus<float>());
			//const float denominator = 1;

			std::cout << "denominator: " << denominator << std::endl;

			probabilities.pt1[x] = 1.0f - constant / denominator;

			const auto functor = Functors::CalculateP1AndPX(cloudBefore[x], probabilities.p, probabilities.p1, probabilities.px, denominator);
			thrust::for_each(thrust::device, idxfirst, idxlast, functor);
			probabilities.error -= std::log(denominator);
		}
		probabilities.error += DIMENSION * cloudBefore.size() * std::log(sigmaSquared) / 2.0f;
	}

	void MStep(		
		const Cloud& cloudBefore,
		const Cloud& cloudAfter,
		const Probabilities& probabilities,
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



		float BeforeCPU[30];
		cudaMemcpy(BeforeCPU, params.beforeT, 30 * sizeof(float), cudaMemcpyDeviceToHost);

		float AfterCPU[30];
		cudaMemcpy(AfterCPU, params.afterT, 30 * sizeof(float), cudaMemcpyDeviceToHost);

		float p1CPU[10];
		cudaMemcpy(p1CPU, params.p1, 10 * sizeof(float), cudaMemcpyDeviceToHost);

		float pt1CPU[10];
		cudaMemcpy(pt1CPU, params.pt1, 10 * sizeof(float), cudaMemcpyDeviceToHost);

		float centerBeforeCPU[3];
		cudaMemcpy(centerBeforeCPU, params.centerBefore, 3 * sizeof(float), cudaMemcpyDeviceToHost);

		float centerAfterCPU[3];
		cudaMemcpy(centerAfterCPU, params.centerAfter, 3 * sizeof(float), cudaMemcpyDeviceToHost);

		float result[9];
		cudaMemcpy(result, params.AMatrix, 9 * sizeof(float), cudaMemcpyDeviceToHost);
		
		float afterTxPX[9];
		cudaMemcpy(afterTxPX, params.afterTxPX, 9 * sizeof(float), cudaMemcpyDeviceToHost);

		float centerBeforexCenterAfter[9];
		cudaMemcpy(centerBeforexCenterAfter, params.centerBeforexCenterAfter, 9 * sizeof(float), cudaMemcpyDeviceToHost);

		printf("np %f\n", Np);

		printf("BeforeCPU\n");
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 10; j++)
			{
				printf("%f ", BeforeCPU[10 * i + j]);
			}
			printf("\n");
		}

		printf("AfterCPU\n");
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 10; j++)
			{
				printf("%f ", AfterCPU[10 * i + j]);
			}
			printf("\n");
		}

		printf("p1CPU\n");
		for (size_t j = 0; j < 10; j++)
		{
			printf("%f ", p1CPU[j]);
		}
		printf("\n");

		printf("pt1CPU\n");
		for (size_t j = 0; j < 10; j++)
		{
			printf("%f ", pt1CPU[j]);
		}
		printf("\n");

		printf("centerBeforeCPU\n");
		for (size_t j = 0; j < 3; j++)
		{
			printf("%f ", centerBeforeCPU[j]);
		}
		printf("\n");

		printf("centerAfterCPU\n");
		for (size_t j = 0; j < 3; j++)
		{
			printf("%f ", centerAfterCPU[j]);
		}
		printf("\n");

		printf("afterTxPX\n");
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				printf("%f ", afterTxPX[3 * j + i]);
			}
			printf("\n");
		}

		printf("centerBeforexCenterAfter\n");
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				printf("%f ", centerBeforexCenterAfter[3 * j + i]);
			}
			printf("\n");
		}

		printf("AMatrix\n");
		for (size_t i = 0; i < 3; i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				printf("%f ", result[3 * j + i]);
			}
			printf("\n");
		}

		printf("matrix U\n");
		Common::PrintMatrix(gU);
		printf("matrix VT\n");
		Common::PrintMatrix(gVT);

		printf("S Matrix\n");
		for (size_t i = 0; i < 3; i++)
		{
			printf("%f ", hostS[i]);
			printf("\n");
		}

		printf("scale numerator %f\n", scaleNumerator);

		/*const Eigen::JacobiSVD<Eigen::MatrixXf> svd = Eigen::JacobiSVD<Eigen::MatrixXf>(AMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);

		const Eigen::Matrix3f matrixU = svd.matrixU();
		const Eigen::Matrix3f matrixV = svd.matrixV();
		const Eigen::Matrix3f matrixVT = matrixV.transpose();

		const Eigen::Matrix3f determinantMatrix = matrixU * matrixVT;

		const Eigen::Matrix3f diag = Eigen::DiagonalMatrix<float, 3>(1.0f, 1.0f, determinantMatrix.determinant());

		const Eigen::Matrix3f EigenRotationMatrix = matrixU * diag * matrixVT;

		const Eigen::Matrix3f EigenScaleNumerator = svd.singularValues().asDiagonal() * diag;

		const float scaleNumerator = EigenScaleNumerator.trace();
		const float sigmaSubtrahend = (EigenBeforeT.transpose().array().pow(2) * probabilities.pt1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterBefore.transpose() * EigenCenterBefore;
		const float scaleDenominator = (EigenAfterT.transpose().array().pow(2) * probabilities.p1.replicate(1, DIMENSION).array()).sum()
			- Np * EigenCenterAfter.transpose() * EigenCenterAfter;

		if (const_scale == false)
		{
			*scale = scaleNumerator / scaleDenominator;
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend - (*scale) * scaleNumerator)) / (float)DIMENSION;
		}
		else
		{
			*sigmaSquared = (InvertedNp * std::abs(sigmaSubtrahend + scaleDenominator - 2 * scaleNumerator)) / (float)DIMENSION;
		}

		const Eigen::Vector3f EigenTranslationVector = EigenCenterBefore - (*scale) * EigenRotationMatrix * EigenCenterAfter;

		*translationVector = ConvertTranslationVector(EigenTranslationVector);

		*rotationMatrix = ConvertRotationMatrix(EigenRotationMatrix);*/
	}

	glm::mat4 CudaCPD(
		const Cloud& cloudBefore,
		const Cloud& cloudAfter,
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
		float sigmaSquared_init = sigmaSquared;

		Probabilities probabilities(cloudBefore.size(), cloudAfter.size());
		CUDAMStepParams mStepParams(cloudBefore.size(), cloudAfter.size(), probabilities);

		if (weight <= 0.0f)
			weight = 1e-6f;
		if (weight >= 1.0f)
			weight = 1.0f - 1e-6f;

		const float constant = (std::pow(2 * M_PI * sigmaSquared, (float)DIMENSION * 0.5f) * weight * cloudAfter.size()) / ((1 - weight) * cloudBefore.size());
		float ntol = tolerance + 10.0f;
		float l = 0.0f;
		Cloud transformedCloud = cloudAfter;
		//EM optimization
		while (*iterations < maxIterations && ntol > tolerance && sigmaSquared > eps)
		{
			//E-step
			if (fgt == Common::ApproximationType::None)
				ComputePMatrix(cloudBefore, transformedCloud, probabilities, constant, sigmaSquared, false, -1.0f);
			//else
			//	probabilities = ComputePMatrixFast(cloudBefore, transformedCloud, constant, weight, &sigmaSquared, sigmaSquared_init, fgt);

			ntol = std::abs((probabilities.error - l) / probabilities.error);
			l = probabilities.error;

			thrust::host_vector<float> p1 = probabilities.p1;
			thrust::host_vector<float> pt1 = probabilities.pt1;
			thrust::host_vector<glm::vec3> px = probabilities.px;

			std::cout << "P1" << std::endl;
			PrintVector(probabilities.p1);
			std::cout << std::endl << "Pt1" << std::endl;
			PrintVector(probabilities.pt1);
			std::cout << std::endl << "PX" << std::endl;
			PrintVector(probabilities.px);

			//M-step
			MStep(cloudBefore, cloudAfter, probabilities, mStepParams, const_scale, &rotationMatrix, &translationVector, &scale, &sigmaSquared);

			//transformedCloud = GetTransformedCloud(cloudAfter, rotationMatrix, translationVector, scale);
			(*error) = sigmaSquared;
			(*iterations)++;
			break;
		}
		//return std::make_pair(scale * rotationMatrix, translationVector);

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
	auto permutedCloud = cloud;// ApplyPermutation(cloud, permutation);
	std::transform(permutedCloud.begin(), permutedCloud.end(), permutedCloud.begin(), [](const Point_f& point) { return Point_f{ point.x * 2.f, point.y * 2.f, point.z * 2.f }; });
	const auto transformedCloud = GetTransformedCloud(cloud, transform);
	const auto transformedPermutedCloud = GetTransformedCloud(permutedCloud, transform);
	timer.StopStage("processing");

	const auto hostBefore = CommonToThrustVector(transformedPermutedCloud);
	const auto hostAfter = CommonToThrustVector(cloud);

	Cloud deviceCloudBefore = hostBefore;
	Cloud deviceCloudAfter = hostAfter;

	timer.StartStage("cpd1");
	const auto icpCalculatedTransform1 = CudaCPD(deviceCloudBefore, deviceCloudAfter, &iterations, &error, testEps, weight, const_scale, max_iterations, testEps, fgt);
	timer.StopStage("cpd1");

	//iterations = 0;
	//error = 1.0f;
	//timer.StartStage("icp2");
	////const auto icpCalculatedTransform2 = CoherentPointDrift::GetRigidCPDTransformationMatrix(cloud, transformedPermutedCloud, &iterations, &error, testEps, weigth, const_scale, max_iterations, testEps, fgt);
	//timer.StopStage("icp2");

	//printf("ICP test (%d iterations) error = %g\n", iterations, error);

	std::cout << "Transform Matrix" << std::endl;
	PrintMatrix(transform);
	//std::cout << "Inverted Transform Matrix" << std::endl;
	//PrintMatrix(glm::inverse(transform));

	//std::cout << "CPD1 Matrix" << std::endl;
	//PrintMatrix(icpCalculatedTransform1.first, icpCalculatedTransform1.second);

	//timer.PrintResults();

	std::cout << "Before" << std::endl;
	PrintVector(deviceCloudBefore);
	std::cout << "After" << std::endl;
	PrintVector(deviceCloudAfter);

	//Common::Renderer renderer(
	//	Common::ShaderType::SimpleModel,
	//	cloud, //red
	//	transformedPermutedCloud, //green
	//	GetTransformedCloud(cloud, icpCalculatedTransform1.first, icpCalculatedTransform1.second), //yellow
	//	//GetTransformedCloud(cloud, icpCalculatedTransform2.first, icpCalculatedTransform2.second)); //blue
	//	std::vector<Point_f>(1)); //green

	//renderer.Show();
}