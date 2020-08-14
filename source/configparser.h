#pragma once
#include "_common.h"
#include <nlohmann/json.hpp>

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

	class ConfigParser
	{
	public:
		ConfigParser(int argc, char** argv);

		ConfigParser(ConfigParser&) = delete;
		ConfigParser(ConfigParser&&) = delete;

		bool IsCorrect() const { return correct; }

		std::string GetBeforeCloudPath() const;
		std::string GetAfterCloudPath() const;

		ComputationMethod GetComputationMethod() const;
		ExecutionPolicy GetExecutionPolicy() const;

		std::vector<int> GetMethodAdditionalParamsInt() const;
		std::vector<float> GetMethodAdditionalParamsFloat() const;

		/// Returns detailed transform given by user
		std::optional<std::pair<glm::mat3, glm::vec3>> GetTransformation() const;

		/// Returns translation, rotation parameter pair
		std::optional<std::pair<float, float>> GetTransformationParameters() const;

		std::optional<int> GetBeforeCloudResize() const;
		std::optional<int> GetAfterCloudResize() const;

	private:

		void LoadConfigFromFile(const std::string& path);
		void ParseMethod(const nlohmann::json& parsed);
		void ParseExecutionPolicy(nlohmann::json parsed);
		void ParseCloudPaths(nlohmann::json parsed);

		bool correct = true;

	};
}
