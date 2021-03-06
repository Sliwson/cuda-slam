#include <Eigen/Dense>

#include "noniterative.h"
#include "timer.h"
#include "configuration.h"
#include "nicputils.h"

#include <thread>

using namespace Common;

namespace NonIterative
{
	std::pair<glm::mat3, glm::vec3> CalculateNonIterativeWithConfiguration(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, Common::Configuration config, int* repetitions, float* error)
	{
		auto maxIterations = config.NicpIterations;

		auto parallel = config.ExecutionPolicy.has_value() ?
			config.ExecutionPolicy.value() == Common::ExecutionPolicy::Parallel :
			true;

		return GetNonIterativeTransformationMatrix(cloudBefore, cloudAfter, repetitions, error, config.ConvergenceEpsilon, maxIterations, config.ApproximationType, parallel, config.NicpSubcloudSize);
	}

	NonIterativeSlamResult GetSingleNonIterativeSlamResult(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter)
	{
		glm::mat3 rotationMatrix = glm::mat3(1.0f);
		glm::vec3 translationVector = glm::vec3(0.0f);
		std::vector<Point_f> transformedCloud = cloudBefore;

		Point_f centerBefore = GetCenterOfMass(cloudBefore);
		Point_f centerAfter = GetCenterOfMass(cloudAfter);

		std::vector<Point_f> alignedBefore = GetAlignedCloud(cloudBefore, centerBefore);
		std::vector<Point_f> alignedAfter = GetAlignedCloud(cloudAfter, centerAfter);

		Eigen::Matrix3Xf matrixBefore = GetMatrix3XFromPointsVector(alignedBefore);
		Eigen::Matrix3Xf matrixAfter = GetMatrix3XFromPointsVector(alignedAfter);

		//const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdBefore = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixBefore, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdBefore = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixBefore, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixBeforeTransposed = svdBefore.matrixU().transpose();

		//const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdAfter = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixAfter, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::JacobiSVD<Eigen::Matrix3Xf> svdAfter = Eigen::JacobiSVD<Eigen::Matrix3Xf>(matrixAfter, Eigen::ComputeThinU | Eigen::ComputeThinV);
		Eigen::Matrix3f uMatrixAfter = svdAfter.matrixU();

		Eigen::Matrix3f rotation = uMatrixAfter * uMatrixBeforeTransposed;

		rotationMatrix = ConvertRotationMatrix(rotation);
		translationVector = glm::vec3(centerAfter) - (rotationMatrix * centerBefore);

		float error = GetMeanSquaredError(alignedBefore, alignedAfter, rotationMatrix);
		return NonIterativeSlamResult(rotationMatrix, translationVector, error);
	}

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrixParallel(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int *repetitions, float* error, float eps, int maxRepetitions, int batchSize, const ApproximationType& calculationType, int subcloudSize)
	{
		int cloudSize = std::min(cloudBefore.size(), cloudAfter.size());
		if (maxRepetitions == -1)
			maxRepetitions = 20;

		// Split number of repetitions to batches
		auto batchesCount = maxRepetitions / batchSize;
		auto lastBatchSize = maxRepetitions % batchSize;
		auto threadsToRun = batchSize;

		std::pair<glm::mat3, glm::vec3> bestTransformation;
		float minError = std::numeric_limits<float>::max();

		// Get subcloud for comparison
		std::vector<Point_f>subcloudVertices = GetSubcloud(cloudBefore, subcloudSize);
		const float maxDistanceForComparison = 1e6;

		// Run NonIterative SLAM for multiple permutations and return the best fit
		std::vector<NonIterativeSlamResult> bestResults;

		std::vector<std::thread> threads(batchSize);
		std::vector<NonIterativeSlamResult> transformationResults(batchSize);
		std::vector<float> errors(batchSize);

		const auto slam_thread_work = [&](int index) {
			const auto permutation = GetRandomPermutationVector(cloudSize);
			const auto permutedBefore = ApplyPermutation(cloudBefore, permutation);
			const auto permutedAfter = ApplyPermutation(cloudAfter, permutation);

			const auto transformationResult = GetSingleNonIterativeSlamResult(permutedBefore, permutedAfter);
			errors[index] = transformationResults[index].getApproximatedError();

			// If not using approximation, calculate error for selected subcloud
			if (calculationType == ApproximationType::None)
			{
				std::vector<Point_f> transformedSubcloud = GetTransformedCloud(subcloudVertices, transformationResult.getRotationMatrix(), transformationResult.getTranslationVector());
				CorrespondingPointsTuple correspondingPoints = GetCorrespondingPoints(transformedSubcloud, permutedAfter, maxDistanceForComparison, true);
				errors[index] = GetMeanSquaredError(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));
			}

			transformationResults[index] = transformationResult;
		};

		for (int i = 0; i <= batchesCount; i++)
		{
			if (i == batchesCount)
			{
				if (lastBatchSize != 0)
					threadsToRun = lastBatchSize;
				else
					break;
			}

			for (int j = 0; j < threadsToRun; j++)
			{
				threads[j] = std::thread(slam_thread_work, j);
			}

			for (int j = 0; j < threadsToRun; j++)
				threads[j].join();

			for (int j = 0; j < threadsToRun; j++)
			{
				*error = errors[j];

				// If not using approximation, calculate error for selected subcloud
				if (calculationType == ApproximationType::None)
				{
					if (*error < minError)
					{
						minError = *error;
						bestTransformation = transformationResults[j].getTransformation();

						if (minError <= eps)
						{
							*repetitions = i * batchSize + j + 1;
							return bestTransformation;
						}
					}
				}
				// If using hybrid approximation, select 5 best fits for further analysis
				else if (calculationType == ApproximationType::Hybrid)
				{
					StoreResultIfOptimal(bestResults, transformationResults[j], 5);
				}
				// If using full approximation, select the best result for further error calculation
				else if (calculationType == ApproximationType::Full)
				{
					StoreResultIfOptimal(bestResults, transformationResults[j], 1);
				}
			}
		}

		std::vector<float> exactErrors(bestResults.size());
		const auto get_exact_error = [&](int index) {
			std::vector<Point_f> transformedSubcloud = GetTransformedCloud(subcloudVertices, bestResults[index].getRotationMatrix(), bestResults[index].getTranslationVector());
			CorrespondingPointsTuple correspondingPoints = GetCorrespondingPoints(transformedSubcloud, cloudAfter, maxDistanceForComparison, true);
			exactErrors[index] = GetMeanSquaredError(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));
		};

		*repetitions = maxRepetitions;

		// If using hybrid approximation, select best result
		if (calculationType == ApproximationType::Full)
		{
			get_exact_error(0);

			*error = exactErrors[0];
			return bestResults[0].getTransformation();
		}
		else if (calculationType == ApproximationType::Hybrid)
		{
			minError = std::numeric_limits<float>::max();
			std::vector<std::thread> errorThreads(bestResults.size());

			for (int i = 0; i < bestResults.size(); i++)
			{
				errorThreads[i] = std::thread(get_exact_error, i);
			}
			
			for (int i = 0; i < bestResults.size(); i++)
				errorThreads[i].join();

			for (int i = 0; i < bestResults.size(); i++)
			{
				*error = exactErrors[i];

				if (*error < minError)
				{
					minError = *error;
					bestTransformation = bestResults[i].getTransformation();

					if (minError <= eps)
					{
						return bestTransformation;
					}
				}
			}
		}

		*error = minError;
		return bestTransformation;
	}

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrixSequential(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int* repetitions, float* error, float eps, int maxRepetitions, const ApproximationType& calculationType, int subcloudSize)
	{
		int cloudSize = std::min(cloudBefore.size(), cloudAfter.size());
		if (maxRepetitions == -1)
			maxRepetitions = 20;

		std::pair<glm::mat3, glm::vec3> bestTransformation;
		float minError = std::numeric_limits<float>::max();

		// Get subcloud for comparison
		std::vector<Point_f>subcloudVertices = GetSubcloud(cloudBefore, subcloudSize);
		const float maxDistanceForComparison = 1e6;

		// Run NonIterative SLAM for multiple permutations and return the best fit
		std::vector<NonIterativeSlamResult> bestResults;
		for (int i = 0; i < maxRepetitions; i++)
		{
			const auto permutation = GetRandomPermutationVector(cloudSize);
			const auto permutedBefore = ApplyPermutation(cloudBefore, permutation);
			const auto permutedAfter = ApplyPermutation(cloudAfter, permutation);

			NonIterativeSlamResult transformationResult = GetSingleNonIterativeSlamResult(permutedBefore, permutedAfter);
			*error = transformationResult.getApproximatedError();
			// If not using approximation, calculate error for selected subcloud
			if (calculationType == ApproximationType::None)
			{
				std::vector<Point_f> transformedSubcloud = GetTransformedCloud(subcloudVertices, transformationResult.getRotationMatrix(), transformationResult.getTranslationVector());
				CorrespondingPointsTuple correspondingPoints = GetCorrespondingPoints(transformedSubcloud, permutedAfter, maxDistanceForComparison, true);
				*error = GetMeanSquaredError(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));

				if (*error < minError)
				{
					minError = *error;
					bestTransformation = transformationResult.getTransformation();

					if (minError <= eps)
					{
						*repetitions = i + 1;
						return bestTransformation;
					}
				}
			}
			// If using hybrid approximation, select 5 best fits for further analysis
			else if (calculationType == ApproximationType::Hybrid)
			{
				StoreResultIfOptimal(bestResults, transformationResult, 5);
			}
			// If using full approximation, select the best result for further error calculation
			else if(calculationType == ApproximationType::Full)
			{
				StoreResultIfOptimal(bestResults, transformationResult, 1);
			}
		}

		*repetitions = maxRepetitions;

		// If using hybrid approximation, select best result
		// If using full approximation, calculate exact error for the best result
		if (calculationType != ApproximationType::None)
		{
			minError = std::numeric_limits<float>::max();
			for (int i = 0; i < bestResults.size(); i++)
			{
				std::vector<Point_f> transformedSubcloud = GetTransformedCloud(subcloudVertices, bestResults[i].getRotationMatrix(), bestResults[i].getTranslationVector());
				CorrespondingPointsTuple correspondingPoints = GetCorrespondingPoints(transformedSubcloud, cloudAfter, maxDistanceForComparison, true);
				*error = GetMeanSquaredError(std::get<0>(correspondingPoints), std::get<1>(correspondingPoints));

				if (*error < minError)
				{
					minError = *error;
					bestTransformation = bestResults[i].getTransformation();

					if (minError <= eps)
					{
						return bestTransformation;
					}
				}
			}
		}

		*error = minError;
		return bestTransformation;
	}

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(const std::vector<Point_f>& cloudBefore, const std::vector<Point_f>& cloudAfter, int *repetitions, float* error, float eps, int maxRepetitions, const ApproximationType& calculationType, bool parallel, int subcloudSize)
	{
		if (parallel)
			return GetNonIterativeTransformationMatrixParallel(cloudBefore, cloudAfter, repetitions, error, eps, maxRepetitions, (int)std::thread::hardware_concurrency(), calculationType, subcloudSize);
		else
			return GetNonIterativeTransformationMatrixSequential(cloudBefore, cloudAfter, repetitions, error, eps, maxRepetitions, calculationType, subcloudSize);
	}
}