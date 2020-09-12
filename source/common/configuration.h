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
		std::optional<std::pair<glm::mat3, glm::vec3>> Transformation = std::nullopt; // rotation matrix, translation vector
		std::optional<std::pair<float, float>> TransformationParameters = std::nullopt; // rotation range, translation range
		std::optional<int> MaxIterations = std::nullopt;
		std::optional<int> CloudBeforeResize = std::nullopt;
		std::optional<int> CloudAfterResize = std::nullopt;
		std::optional<float> CloudSpread = std::nullopt;
		std::optional<int> RandomSeed = std::nullopt;
		std::optional<float> NoiseAffectedPointsBefore = std::nullopt;
		std::optional<float> NoiseAffectedPointsAfter = std::nullopt;

		//optional parameters with default values
		bool ShowVisualisation = false;
		float MaxDistanceSquared = 1000.f;
		ApproximationType ApproximationType = ApproximationType::Hybrid;
		int NicpBatchSize = 16;
		int NicpIterations = 32;
		int NicpSubcloudSize = 1000;
		float CpdWeight = .3f;
		bool CpdConstScale = true;
		float CpdTolerance = 1e-3f;
		float ConvergenceEpsilon = 1e-3f;
		float NoiseIntensityBefore = 0.1f;
		float NoiseIntensityAfter = 0.1f;
		int AdditionalOutliersBefore = 0;
		int AdditionalOutliersAfter = 0;
		float RatioOfFarField = 10.0f;
		int OrderOfTruncation = 8;

		void Print();
	};
}
