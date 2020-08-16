#pragma once
#include "_common.h"

namespace Common
{
	enum class ComputationMethod
	{
		Icp,
		NoniterativeIcp,
		Cpd
	};

	enum class ExecutionPolicy
	{
		Sequential,
		Parallel
	};

	struct Configuration
	{
		//required parameters
		ComputationMethod ComputationMethod = ComputationMethod::Icp;
		std::string BeforePath;
		std::string AfterPath;

		//optional parameters
		std::optional<ExecutionPolicy> ExecutionPolicy = std::nullopt;
		std::optional<std::pair<glm::mat3, glm::vec3>> Transformation = std::nullopt;
		std::optional<std::pair<float, float>> TransformationParameters = std::nullopt;
		std::optional<int> MaxIterations = std::nullopt;

		void Print();
	};
}
