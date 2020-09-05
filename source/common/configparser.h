#pragma once
#include "_common.h"
#include "configuration.h"
#include <nlohmann/json.hpp>

namespace Common 
{
	class ConfigParser
	{
	public:
		ConfigParser(int argc, char** argv);

		bool IsCorrect() const { return correct; }
		Configuration GetConfiguration() const { return config; }

	private:

		void LoadConfigFromFile(const std::string& path);
		void ParseMethod(const nlohmann::json& parsed);
		void ParseCloudPaths(const nlohmann::json& parsed);
		void ParseExecutionPolicy(const nlohmann::json& parsed);
		void ParseTransformation(const nlohmann::json& parsed);
		void ParseTransformationParameters(const nlohmann::json& parsed);
		void ParseAdditionalParameters(const nlohmann::json& parsed);

		void ValidateConfiguration();

		template<typename T>
		std::optional<T> ParseRequired(const nlohmann::json& parsed, std::string name);
		template<typename T>
		std::optional<T> ParseOptional(const nlohmann::json& parsed, std::string name);
		template<typename T>
		T ParseOptional(const nlohmann::json& parsed, std::string name, T defaultValue);

		bool correct = true;
		Configuration config;
	};

	template<typename T>
	inline std::optional<T> ConfigParser::ParseOptional(const nlohmann::json& parsed, std::string name)
	{
		if (parsed.find(name) != parsed.end())
		{
			auto prop = parsed[name];
			return prop.get<T>();
		}
		else
		{
			return std::nullopt;
		}
	}

	template<typename T>
	inline T ConfigParser::ParseOptional(const nlohmann::json& parsed, std::string name, T defaultValue)
	{
		auto opt = ParseOptional<T>(parsed, name);
		if (opt != std::nullopt)
			return opt.value();

		return defaultValue;
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
