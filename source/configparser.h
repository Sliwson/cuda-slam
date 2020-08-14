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

		bool IsCorrect() const { return correct; }

		std::string GetBeforeCloudPath() const { return beforePath; }
		std::string GetAfterCloudPath() const { return afterPath; }

		ComputationMethod GetComputationMethod() const { return computationMethod; }
		std::optional<ExecutionPolicy> GetExecutionPolicy() const { return executionPolicy; }

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
		void ParseCloudPaths(const nlohmann::json& parsed);
		void ParseExecutionPolicy(const nlohmann::json& parsed);
		void ParseTransformation(const nlohmann::json& parsed);
		void ParseTransformationParameters(const nlohmann::json& parsed);

		void ValidateConfiguration();

		template<typename T>
		std::optional<T> ParseRequired(const nlohmann::json& parsed, std::string name);
		template<typename T>
		std::optional<T> ParseOptional(const nlohmann::json& parsed, std::string name);

		bool correct = true;
		ComputationMethod computationMethod = ComputationMethod::Icp;
		std::optional<ExecutionPolicy> executionPolicy = std::nullopt;
		std::string beforePath;
		std::string afterPath;
		std::optional<std::pair<glm::mat3, glm::vec3>> transformation;
		std::optional<std::pair<float, float>> transformationParameters;
	};

	template<typename T>
	inline std::optional<T> ConfigParser::ParseOptional(const nlohmann::json& parsed, std::string name)
	{
		try
		{
			auto prop = parsed[name];
			return prop.get<T>();
		}
		catch (...)
		{
			return std::nullopt;
		}
	}

	template<typename T>
	inline std::optional<T> ConfigParser::ParseRequired(const nlohmann::json& parsed, std::string name)
	{
		auto prop = ParseOptional<T>(parsed, name);
		if (!prop.has_value())
		{
			printf("Parsing error: No required parameter '%s'\n", name.c_str());
			correct = false;
		}

		return prop;
	}
}
