#pragma once
#include "_common.h"
#include <optional>

namespace Common
{
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
		std::optional<int> CloudResize = std::nullopt;

		//optional parameters with default values
		bool ShowVisualisation = false;
		float MaxDistanceSquared = 1.f;
		ApproximationType ApproximationType = ApproximationType::Hybrid;
		float CpdWeight = .3f;

		void Print();
	};
}
