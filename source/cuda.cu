#include "cuda.cuh"
#include "functors.cuh"
#include "svdparams.cuh"
#include "timer.h"
#include "testutils.h"

namespace {
	typedef thrust::device_vector<glm::vec3> Cloud;
	typedef thrust::device_vector<int> IndexIterator;
	typedef thrust::permutation_iterator<Cloud, IndexIterator> Permutation;

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

	thrust::host_vector<glm::vec3> CommonToThrustVector(const std::vector<Common::Point_f>& vec)
	{
		thrust::host_vector<glm::vec3> hostCloud(vec.size());
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

	__device__ float GetDistanceSquared(const glm::vec3& first, const glm::vec3& second)
	{
		const auto d = second - first;
		return d.x * d.x + d.y * d.y + d.z * d.z;
	}

	__global__ void FindCorrespondences(int* result, const glm::vec3* before, const glm::vec3* after, int beforeSize, int afterSize)
	{
		int targetIdx = blockDim.x * blockIdx.x + threadIdx.x;
		if (targetIdx < beforeSize)
		{
			const glm::vec3 vector = before[targetIdx];
			int nearestIdx = 0;
			float smallestError = GetDistanceSquared(vector, after[0]);
			for (int i = 1; i < afterSize; i++)
			{
				const auto dist = GetDistanceSquared(vector, after[i]);
				if (dist < smallestError)
				{
					smallestError = dist;
					nearestIdx = i;
				}
			}

			result[targetIdx] = nearestIdx;
		}
	}

	void GetCorrespondingPoints(thrust::device_vector<int>& indices, const Cloud& before, const Cloud& after)
	{
#ifdef USE_CORRESPONDENCES_KERNEL
		int* dIndices = thrust::raw_pointer_cast(indices.data());
		const glm::vec3* dBefore = thrust::raw_pointer_cast(before.data());
		const glm::vec3* dAfter = thrust::raw_pointer_cast(after.data());
		int beforeSize = before.size();
		int afterSize = after.size();

		constexpr int threadsPerBlock = 256;
		const int blocksPerGrid = (beforeSize + threadsPerBlock - 1) / threadsPerBlock;
		FindCorrespondences << <blocksPerGrid, threadsPerBlock >> > (dIndices, dBefore, dAfter, beforeSize, afterSize);
		cudaDeviceSynchronize();
#else
		const auto nearestFunctor = Functors::FindNearestIndex(after);
		thrust::transform(thrust::device, before.begin(), before.end(), indices.begin(), nearestFunctor);
#endif
	}

	float GetMeanSquaredError(const IndexIterator& permutation, const Cloud& before, const Cloud& after)
	{
		auto permutationIteratorBegin = thrust::make_permutation_iterator(after.begin(), permutation.begin());
		auto permutationIteratorEnd = thrust::make_permutation_iterator(after.end(), permutation.end());
		auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(permutationIteratorBegin, before.begin()));
		auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(permutationIteratorEnd, before.end()));
		auto mseFunctor = Functors::MSETransform();
		auto sumFunctor = thrust::plus<float>();
		auto result = thrust::transform_reduce(thrust::device, zipBegin, zipEnd, mseFunctor, 0.f, sumFunctor);
		return result / after.size();
	}

	void GetAlignedCloud(const Cloud& source, Cloud& target)
	{
		const auto centroid = CalculateCentroid(source);
		const auto transform = Functors::TranslateTransform(-centroid);
		thrust::transform(thrust::device, source.begin(), source.end(), target.begin(), transform);
	}

	void CuBlasMultiply(float* A, float* B, float* C, int size, CudaSvdParams& params)
	{
		const float alpha = 1.f, beta = 0.f;
		cublasSgemm(params.multiplyHandle, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, size, &alpha, A, 3, B, size, &beta, C, 3);
	}

	glm::mat3 CreateGlmMatrix(float* squareMatrix)
	{
		return glm::transpose(glm::make_mat3(squareMatrix));
	}

	glm::mat4 LeastSquaresSVD(const IndexIterator& permutation, const Cloud& before, const Cloud& after, Cloud& alignBefore, Cloud& alignAfter, CudaSvdParams params)
	{
		const int size = before.size();

		//align arrays
		const auto centroidBefore = CalculateCentroid(before);
		GetAlignedCloud(before, alignBefore);

		auto permutationIteratorBegin = thrust::make_permutation_iterator(after.begin(), permutation.begin());
		auto permutationIteratorEnd = thrust::make_permutation_iterator(after.end(), permutation.end());
		thrust::copy(thrust::device, permutationIteratorBegin, permutationIteratorEnd, alignAfter.begin());
		const auto centroidAfter = CalculateCentroid(alignAfter);
		GetAlignedCloud(alignAfter, alignAfter);

		//create array AFTER (transposed)
		auto countingBegin = thrust::make_counting_iterator<int>(0);
		auto countingEnd = thrust::make_counting_iterator<int>(alignAfter.size());
		auto zipBegin = thrust::make_zip_iterator(thrust::make_tuple(countingBegin, alignAfter.begin()));
		auto zipEnd = thrust::make_zip_iterator(thrust::make_tuple(countingEnd, alignAfter.end()));

		auto convertAfter = Functors::GlmToCuBlas(true, size, params.workAfter);
		thrust::for_each(thrust::device, zipBegin, zipEnd, convertAfter);

		//create array BEFORE
		const auto beforeZipBegin = thrust::make_zip_iterator(thrust::make_tuple(countingBegin, alignBefore.begin()));
		const auto beforeZipEnd = thrust::make_zip_iterator(thrust::make_tuple(countingEnd, alignBefore.end()));
		auto convertBefore = Functors::GlmToCuBlas(false, before.size(), params.workBefore);
		thrust::for_each(thrust::device, beforeZipBegin, beforeZipEnd, convertBefore);

		//multiply
		CuBlasMultiply(params.workBefore, params.workAfter, params.multiplyResult, size, params);
		float result[9];
		cudaMemcpy(result, params.multiplyResult, 9 * sizeof(float), cudaMemcpyDeviceToHost);
		auto matrix = CreateGlmMatrix(result);
		//return Common::GetTransform(matrix, centroidBefore, centroidAfter);

		float transposed[9];
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				transposed[3 * i + j] = result[3 * j + i];
		cudaMemcpy(params.multiplyResult, transposed, 9 * sizeof(float), cudaMemcpyHostToDevice);

		//svd
		cusolverDnSgesvd(params.solverHandle, 'A', 'A', 3, 3, params.multiplyResult, 3, params.S, params.U, 3, params.VT, 3, params.work, params.workSize, nullptr, params.devInfo);
		int svdResultInfo = 0;
		cudaMemcpy(&svdResultInfo, params.devInfo, sizeof(int), cudaMemcpyDeviceToHost);
		if (svdResultInfo != 0)
			printf("Svd execution failed!\n");

		float hostS[9], hostVT[9], hostU[9];
		const int copySize = 9 * sizeof(float);
		cudaMemcpy(hostS, params.S, copySize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostVT, params.VT, copySize, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostU, params.U, copySize, cudaMemcpyDeviceToHost);

		auto gVT = glm::transpose(CreateGlmMatrix(hostVT));
		auto gU = glm::transpose(CreateGlmMatrix(hostU));

		//revert signs to match svd cpu solution
		for (int i = 0; i < 3; i++)
		{
			gU[1][i] = -gU[1][i];
			gVT[i][1] = -gVT[i][1];
		}

		const float determinant = glm::determinant(gU * gVT);
		const auto diagonal = glm::diagonal3x3(glm::vec3{ 1.f, 1.f, determinant });
		const auto rotation = gU * diagonal * gVT;

		const auto translation = centroidAfter - rotation * centroidBefore;

		auto transformation = glm::mat4(0.f);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				transformation[i][j] = rotation[i][j];

		transformation[3][0] = translation.x;
		transformation[3][1] = translation.y;
		transformation[3][2] = translation.z;
		transformation[3][3] = 1.0f;

		return transformation;
	}

	glm::mat4 CudaICP(const Cloud& before, const Cloud& after)
	{
		const int maxIterations = 60;
		const float TEST_EPS = 1e-5;
		float previousError = std::numeric_limits<float>::max();

		int iterations = 0;
		glm::mat4 transformationMatrix(1.0f);
		glm::mat4 previousTransformationMatrix = transformationMatrix;

		//do not change before vector - copy it for calculations
		const int size = std::max(before.size(), after.size());
		Cloud workingBefore(size);
		Cloud alignBefore(size);
		Cloud alignAfter(size);
		thrust::device_vector<int> indices(before.size());
		thrust::copy(thrust::device, before.begin(), before.end(), workingBefore.begin());

		//allocate memory for cuBLAS
		CudaSvdParams params(size, size);

		while (iterations < maxIterations)
		{
			GetCorrespondingPoints(indices, workingBefore, after);

			transformationMatrix = LeastSquaresSVD(indices, workingBefore, after, alignBefore, alignAfter, params) * transformationMatrix;

			TransformCloud(before, workingBefore, transformationMatrix);
			float error = GetMeanSquaredError(indices, workingBefore, after);
			printf("Iteration: %d, error: %f\n", iterations, error);
			if (error < TEST_EPS)
				break;

			if (error > previousError)
			{
				printf("Error has increased, aborting\n");
				transformationMatrix = previousTransformationMatrix;
				break;
			}

			previousTransformationMatrix = transformationMatrix;
			previousError = error;
			iterations++;
		}

		params.Free();
		return transformationMatrix;
	}

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

	//Probabilities ComputePMatrix(
	//	const Cloud& cloudBefore,
	//	const Cloud& cloudTransformed,
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
		const Cloud& cloudBefore,
		const Cloud& cloudAfter,
		int* iterations,
		float* error,
		float eps,
		float weight,
		bool const_scale,
		int maxIterations,
		float tolerance,
		FastGaussTransform::FGTType fgt)
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
			//Cloud transformedCloud = cloudAfter;
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

	void CorrespondencesTest()
	{
		const int size = 100;
		thrust::device_vector<glm::vec3> input(size);
		thrust::device_vector<glm::vec3> output(size);
		thrust::device_vector<int> result(size);

		for (int i = 0; i < size; i++)
		{
			const auto vec = glm::vec3(i);
			input[i] = vec;
			output[size - i - 1] = vec;
		}

		GetCorrespondingPoints(result, input, output);
		thrust::host_vector<int> copy = result;
		bool ok = true;
		int hostArray[size];
		for (int i = 0; i < size; i++)
		{
			hostArray[i] = copy[i];
			if (copy[i] != size - i - 1)
				ok = false;
		}

		printf("Correspondence test [%s]\n", ok ? "OK" : "FAILED");
	}

	void MultiplicationTest()
	{
		const int size = 100;

		float ones[3 * size];
		for (int i = 0; i < 3 * size; i++)
			ones[i] = 1.f;

		CudaSvdParams params(size, size);
		cudaMemcpy(params.workBefore, ones, 3 * size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(params.workAfter, ones, 3 * size * sizeof(float), cudaMemcpyHostToDevice);

		CuBlasMultiply(params.workBefore, params.workAfter, params.multiplyResult, size, params);

		float result[9];
		cudaMemcpy(result, params.multiplyResult, 9 * sizeof(float), cudaMemcpyDeviceToHost);

		bool ok = true;
		for (int i = 0; i < 9; i++)
			if (abs(result[i] - size) > 1e-5)
				ok = false;

		printf("Multiplication test [%s]\n", ok ? "OK" : "FAILED");
		params.Free();
	}
}

void CudaTest()
{
	/****************************/
	//TESTS
	/****************************/
	CorrespondencesTest();
	MultiplicationTest();

	/****************************/
	//ALGORITHM
	/****************************/
	const auto testCloud = LoadCloud("data/rose.obj");
	const auto testCorrupted = LoadCloud("data/rose.obj");

	const auto hostBefore = CommonToThrustVector(testCloud);
	const auto hostAfter = CommonToThrustVector(testCorrupted);

	Cloud deviceCloudBefore = hostBefore;
	Cloud deviceCloudAfter = hostAfter;

	Cloud calculatedCloud(hostAfter.size());

	const auto scaleInput = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudBefore.begin(), deviceCloudBefore.end(), deviceCloudBefore.begin(), scaleInput);
	const auto scaleInputCorrupted = Functors::ScaleTransform(1000.f);
	thrust::transform(thrust::device, deviceCloudAfter.begin(), deviceCloudAfter.end(), deviceCloudAfter.begin(), scaleInputCorrupted);

	const auto sampleTransform = glm::rotate(glm::translate(glm::mat4(1), { 0.05f, 0.05f, 0.05f }), glm::radians(5.f), { 0.5f, 0.5f, 0.5f });
	TransformCloud(deviceCloudAfter, deviceCloudAfter, sampleTransform);

	auto start = std::chrono::high_resolution_clock::now();
	const auto result = CudaICP(deviceCloudBefore, deviceCloudAfter);
	auto stop = std::chrono::high_resolution_clock::now();
	printf("Size: %d points, duration: %dms\n", testCloud.size(), std::chrono::duration_cast<std::chrono::milliseconds>(stop - start));

	TransformCloud(deviceCloudBefore, calculatedCloud, result);

	Common::Renderer renderer(
		Common::ShaderType::SimpleModel,
		ThrustToCommonVector(deviceCloudBefore), //grey
		ThrustToCommonVector(deviceCloudAfter), //blue
		ThrustToCommonVector(calculatedCloud), //red
		std::vector<Point_f>(1));

	renderer.Show();
}

void CPDTest()
{
	const char* objectPath = "data/bunny.obj";
	const int pointCount = 10;
	const float testEps = 1e-6f;
	const float weight = 0.0f;
	const bool const_scale = false;
	const int max_iterations = 50;
	const FastGaussTransform::FGTType fgt = FastGaussTransform::FGTType::None;

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