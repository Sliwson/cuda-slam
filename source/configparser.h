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

		std::string GetBeforeCloudPath() const { return beforePath; }
		std::string GetAfterCloudPath() const { return afterPath; }

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
		void ParseCloudPaths(const nlohmann::json& parsed);
		void ParseExecutionPolicy(const nlohmann::json& parsed);

		template<typename T>
		std::optional<T> ParseRequired(const nlohmann::json& parsed, std::string name);
		template<typename T>
		std::optional<T> ParseOptional(const nlohmann::json& parsed, std::string name);

		bool correct = true;
		ComputationMethod computationMethod = ComputationMethod::Icp;
		std::string beforePath;
		std::string afterPath;

	};

	template<typename T>
	inline std::optional<T> ConfigParser::ParseOptional(const nlohmann::json& parsed, std::string name)
	{
		auto prop = parsed[name];
		try
		{
			return prop.get<T>();
		}
		catch 
		{
			return std::nullpopt;
		}
	}

	template<typename T>
	inline std::optional<T> ConfigParser::ParseRequired(const nlohmann::json& parsed, std::string name)
	{
		auto prop = ParseOptional<T>(parsed, name);
		if (!method.has_value())
		{
			printf("Parsing error: No required parameter '%s'\n", name.c_str());
			correct = false;
		}

		return prop;
	}
}
