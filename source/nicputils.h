#pragma once

#include "common.h"

namespace Common
{
	class NonIterativeSlamResult
	{
	public:
		NonIterativeSlamResult() {}
		NonIterativeSlamResult(const glm::mat3& rotationMatrix, const glm::vec3& translationVector, const float& approximatedError)
			:rotationMatrix(rotationMatrix), translationVector(translationVector), approximatedError(approximatedError) {}

		glm::mat3 getRotationMatrix() const { return rotationMatrix; }
		glm::vec3 getTranslationVector() const { return translationVector; }
		float getApproximatedError() const { return approximatedError; }
		std::pair<glm::mat3, glm::vec3> getTransformation() const { return std::make_pair(rotationMatrix, translationVector); }
		glm::mat4 getTransformationMatrix() const { return ConvertToTransformationMatrix(rotationMatrix, translationVector); }

	private:
		glm::mat3 rotationMatrix;
		glm::vec3 translationVector;
		float approximatedError;
	};

	void StoreResultIfOptimal(std::vector<NonIterativeSlamResult>& results, const NonIterativeSlamResult& newResult, int desiredLength);
}