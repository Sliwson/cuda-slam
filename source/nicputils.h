#pragma once

#include "_common.h"

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
		glm::mat4 getTransformationMatrix() const { auto result = glm::mat4(rotationMatrix); result[3] = glm::vec4(translationVector, 1.f); return result; }

	private:
		glm::mat3 rotationMatrix;
		glm::vec3 translationVector;
		float approximatedError;
	};

	void StoreResultIfOptimal(std::vector<NonIterativeSlamResult>& results, const NonIterativeSlamResult& newResult, int desiredLength);
}