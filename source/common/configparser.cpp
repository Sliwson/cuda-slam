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
			ParseTransformation(parsed);
			ParseTransformationParameters(parsed);
			ParseAdditionalParameters(parsed);

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
		auto translationIt = parsed.find("translation");
		auto rotationIt = parsed.find("rotation");
		auto scale = ParseOptional(parsed, "scale", 1.0f);

		if (translationIt != parsed.end() && rotationIt != parsed.end())
		{
			auto translation = parsed["translation"];
			auto rotation = parsed["rotation"];

			if (translation.size() != 3 || rotation.size() != 9)
			{
				printf("Parsing error: Wrong translation or rotation size\n");
				correct = false;
				return;
			}

			try 
			{
				glm::mat3 rotationMatrix;
				for (int x = 0; x < 3; x++)
					for (int y = 0; y < 3; y++)
						rotationMatrix[y][x] = rotation[x * 3 + y].get<float>();

				glm::vec3 translationVector;
				for (int i = 0; i < 3; i++)
					translationVector[i] = translation[i].get<float>();

				config.Transformation = std::make_pair(scale * rotationMatrix, translationVector);
			}
			catch (...)
			{
				printf("Parsing error: Error parsing translation or rotation parameter\n");
				correct = false;
			}
		}
	}

	void ConfigParser::ParseTransformationParameters(const nlohmann::json& parsed)
	{
		auto translationIt = parsed.find("translation-range");
		auto rotationIt = parsed.find("rotation-range");

		if (translationIt != parsed.end() && rotationIt != parsed.end())
		{
			try 
			{
				auto translationRange = parsed["translation-range"];
				auto rotationRange = parsed["rotation-range"];

				auto translationRangeValue = translationRange.get<float>();
				auto rotationRangeValue = rotationRange.get<float>();

				config.TransformationParameters = std::make_pair(rotationRangeValue, translationRangeValue);
			}
			catch (...)
			{
				printf("Parsing error: Error parsing translation-range or rotation-range parameter\n");
				correct = false;
			}
		}
	}

	void ConfigParser::ParseAdditionalParameters(const nlohmann::json& parsed)
	{
		config.MaxIterations = ParseOptional<int>(parsed, "max-iterations");

		config.CloudBeforeResize = ParseOptional<int>(parsed, "cloud-before-resize");

		config.CloudAfterResize = ParseOptional<int>(parsed, "cloud-after-resize");

		config.CloudSpread = ParseOptional<float>(parsed, "cloud-spread");

		config.RandomSeed = ParseOptional<int>(parsed, "random-seed");

		config.NoiseAffectedPointsBefore = ParseOptional<float>(parsed, "noise-affected-points-before");

		config.NoiseAffectedPointsAfter = ParseOptional<float>(parsed, "noise-affected-points-after");

		config.ShowVisualisation = ParseOptional(parsed, "show-visualisation", false);

		config.MaxDistanceSquared = ParseOptional(parsed, "max-distance-squared", 1000.f);

		config.CpdWeight = ParseOptional<float>(parsed, "cpd-weight", .3f);

		config.ApproximationType = [this, &parsed]() {
			auto approximationType = ParseOptional<std::string>(parsed, "approximation-type");
			if (!approximationType.has_value())
				return ApproximationType::Hybrid;

			const std::map<std::string, ApproximationType> mapping = {
				{ "full", ApproximationType::Full },
				{ "hybrid", ApproximationType::Hybrid },
				{ "none", ApproximationType::None }
			};

			const auto approximationString = approximationType.value();
			if (auto result = mapping.find(approximationString); result != mapping.end())
				return result->second;
			else
				return ApproximationType::Hybrid;
		}();

		config.NicpBatchSize = ParseOptional(parsed, "nicp-batch-size", 16);

		config.NicpIterations = ParseOptional(parsed, "nicp-iterations", 32);

		config.NicpSubcloudSize = ParseOptional(parsed, "nicp-subcloud-size", 1000);
		
		config.CpdWeight = ParseOptional(parsed, "cpd-weight", 0.3f);
		
		config.CpdConstScale = ParseOptional(parsed, "cpd-const-scale", false);
		
		config.CpdTolerance = ParseOptional(parsed, "cpd-tolerance", 1e-3);

		config.ConvergenceEpsilon = ParseOptional(parsed, "convergence-epsilon", 1e-3);

		config.NoiseIntensityBefore = ParseOptional(parsed, "noise-intensity-before", 0.1f);

		config.NoiseIntensityAfter = ParseOptional(parsed, "noise-intensity-after", 0.1f);

		config.AdditionalOutliersBefore = ParseOptional(parsed, "additional-outliers-before", 0);

		config.AdditionalOutliersAfter = ParseOptional(parsed, "additional-outliers-after", 0);

		config.RatioOfFarField = ParseOptional(parsed, "fgt-ratio-of-far-field", 10.0f);

		config.OrderOfTruncation = ParseOptional(parsed, "fgt-order-of-truncation", 8);
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
