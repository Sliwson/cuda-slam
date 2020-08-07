#include "configparser.h"

namespace {
	constexpr const char* DEFAULT_PATH = "configuration/default.json";
}

ConfigParser::ConfigParser(int argc, char** argv)
{
	const std::string defaultPath = { DEFAULT_PATH };
	if (argc == 0)
	{
		printf("No config passed, loading: %s\n", DEFAULT_PATH);
		LoadConfigFromFile(defaultPath);
	}
	else if (argc == 1)
	{
		const std::string path = { argv[0] };
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
}
