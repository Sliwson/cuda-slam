#pragma once
#include <utility>
#include <tuple>
#include "common.h"
#include "configuration.h"

using namespace Common;

namespace NonIterative
{
	class NonIterativeSlamResult
	{
	private:
		glm::mat3 rotationMatrix;
		glm::vec3 translationVector;
		std::vector<Point_f> cloudAfter;
		float approximatedError;

	public:
		NonIterativeSlamResult()
		{
		}

		NonIterativeSlamResult(const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const std::vector<Point_f>& cloudAfter, const float& approximatedError)
			:rotationMatrix(rotationMatrix), translationVector(translationVector), cloudAfter(cloudAfter), approximatedError(approximatedError)
		{
		}

		glm::mat3 getRotationMatrix() const
		{
			return rotationMatrix;
		}

		glm::vec3 getTranslationVector() const
		{
			return translationVector;
		}

		std::vector<Point_f> getCloudAfter() const
		{
			return cloudAfter;
		}

		float getApproximatedError() const
		{
			return approximatedError;
		}

		std::pair<glm::mat3, glm::vec3> getTransformation() const
		{
			return std::make_pair(rotationMatrix, translationVector);
		}
	};



	std::pair<glm::mat3, glm::vec3> CalculateNonIterativeWithConfiguration(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter,
		Common::Configuration config);

	NonIterativeSlamResult GetSingleNonIterativeSlamResult(
		const std::vector<Point_f>& cloudBefore,
		const std::vector<Point_f>& cloudAfter);

	std::pair<glm::mat3, glm::vec3> GetNonIterativeTransformationMatrix(
		const std::vector<Point_f>& cloudBefore, 
		const std::vector<Point_f>& cloudAfter, 
		float* error, 
		float eps, 
		int maxRepetitions, 
		const ApproximationType& calculationType, 
		bool parallel = false,
		int subcloudSize = -1);

	void StoreResultIfOptimal(std::vector<NonIterativeSlamResult>& results, const NonIterativeSlamResult& newResult, int desiredLength);
}