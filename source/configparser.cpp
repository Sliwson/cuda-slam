#include "configparser.h"

#include <fstream>

namespace {
	constexpr const char* DEFAULT_PATH = "config/default.json";
}

namespace Common
{
	ConfigParser::ConfigParser(int argc, char** argv)
	{
		const std::string defaultPath = { DEFAULT_PATH };
		if (argc == 1)
		{
			printf("No config passed, loading: %s\n", DEFAULT_PATH);
			LoadConfigFromFile(defaultPath);
		}
		else if (argc == 2)
		{
			const std::string path = { argv[1] };
			if (std::filesystem::exists(path))
			{
				printf("Loading config from: %s\n", path.c_str());
				LoadConfigFromFile(path);
			}
			else
			{
				printf("File: %s does not exist, loading default config\n", path.c_str());
				LoadConfigFromFile(defaultPath);
			}
		}
		else
		{
			printf("Usage: %s (config_path)\n", argv[0]);
			printf("Loading default config\n");
			LoadConfigFromFile(defaultPath);
		}
	}

	void ConfigParser::LoadConfigFromFile(const std::string& path)
	{
		auto stream = std::ifstream(path);
		auto content = std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
		stream.close();

		try {
			auto parsed = nlohmann::json::parse(content);
			ParseMethod(parsed);
			ParseCloudPaths(parsed);
			ParseExecutionPolicy(parsed);

			ValidateConfiguration();
		}
		catch (std::exception ex)
		{
			printf("Parsing error: %s\n", ex.what());
			correct = false;
		}
		catch (...)
		{
			printf("Parsing error: unknown exception\n");
			correct = false;
		}
	}

	void ConfigParser::ParseMethod(const nlohmann::json& parsed)
	{
		auto method = ParseRequired<std::string>(parsed, "method");
		if (!method.has_value())
			return;

		const std::map<std::string, ComputationMethod> mapping = {
			{ "icp", ComputationMethod::Icp },
			{ "nicp", ComputationMethod::NoniterativeIcp },
			{ "cpd", ComputationMethod::Cpd }
		};

		const auto methodStr = method.value();
		if (auto result = mapping.find(methodStr); result != mapping.end())
		{
			config.ComputationMethod = result->second;
		}
		else
		{
			printf("Parsing error: Computational method %s not supported\n", methodStr.c_str());
			correct = false;
		}
	}
		
	void ConfigParser::ParseCloudPaths(const nlohmann::json& parsed)
	{
		auto beforePathOpt = ParseRequired<std::string>(parsed, "before-path");
		auto afterPathOpt = ParseRequired<std::string>(parsed, "after-path");
		if (!beforePathOpt.has_value() || !afterPathOpt.has_value())
			return;
	
		config.BeforePath = beforePathOpt.value();
		config.AfterPath = afterPathOpt.value();
	}

	void ConfigParser::ParseExecutionPolicy(const nlohmann::json& parsed)
	{
		auto method = ParseOptional<std::string>(parsed, "policy");
		if (!method.has_value())
			return;

		const std::map<std::string, ExecutionPolicy> mapping = {
			{ "parallel", ExecutionPolicy::Parallel },
			{ "sequential", ExecutionPolicy::Sequential }
		};

		const auto methodStr = method.value();
		if (auto result = mapping.find(methodStr); result != mapping.end())
		{
			config.ExecutionPolicy = result->second;
		}
		else
		{
			printf("Parsing warning: Execution policy %s not supported\n", methodStr.c_str());
			correct = false;
		}
	}

	void ConfigParser::ParseTransformation(const nlohmann::json& parsed)
	{
		bool loadedTransformation = false;
		try
		{
			auto translation = parsed["translation"];
			auto rotation = parsed["rotation"];
			loadedTransformation = true;

			if (translation.size() != 3 || rotation.size() != 9)
			{
				printf("Parsing error: Wrong translation or rotation size\n");
				correct = false;
				return;
			}

			glm::mat3 rotationMatrix;
			for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
					rotationMatrix[y][x] = rotation[y * 3 + x].get<float>();

			glm::vec3 translationVector;
			for (int i = 0; i < 3; i++)
				translationVector[i] = translation[i].get<float>();

			config.Transformation = std::make_pair(rotationMatrix, translationVector);
		}
		catch (...) 
		{
			if (loadedTransformation)
			{
				printf("Parsing error: Error parsing translation or rotation parameter\n");
				correct = false;
			}
		}
	}

	void ConfigParser::ValidateConfiguration()
	{
		if (!config.Transformation.has_value() && !config.TransformationParameters.has_value())
		{
			printf("Parsing error: transformation or transformation parameters have to be provided\n");
			correct = false;
		}
	}
}
