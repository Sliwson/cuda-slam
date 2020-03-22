#include "shaderfactory.h"
#include "shadertype.h"

namespace Common
{
	namespace shaderPath
	{
		const char* vertexShaderPath = "source/shaders/vertexshader.vert";
		const char* fragmentShaderPath = "source/shaders/fragmentshader.frag";
	}

	ShaderFactory::ShaderFactory()
	{
		simpleModel = std::make_shared<Shader>(shaderPath::vertexShaderPath, shaderPath::fragmentShaderPath);
	}

	ShaderFactory& ShaderFactory::getInstance()
	{
		static ShaderFactory instance;

		return instance;
	}

	std::shared_ptr<Shader> ShaderFactory::getShader(ShaderType type)
	{
		switch (type)
		{
		case ShaderType::simpleModel:
			return simpleModel;
		default:
			break;
		}
		return simpleModel;
	}
}