#include "shaderfactory.h"

namespace Common
{
	namespace shaderPath
	{
		const char* vertexShaderPath = "source/shaders/vertexshader.vert";
		const char* fragmentShaderPath = "source/shaders/fragmentshader.frag";
	}

	ShaderFactory::ShaderFactory() :
		simpleModel(shaderPath::vertexShaderPath, shaderPath::fragmentShaderPath)
	{

	}

	ShaderFactory& ShaderFactory::getInstance()
	{
		static ShaderFactory instance;

		return instance;
	}

	Shader& ShaderFactory::getShader(ShaderType type)
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