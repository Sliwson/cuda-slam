#include "shaderfactory.h"
#include "shadertype.h"

namespace Common
{
	ShaderFactory::ShaderFactory()
	{
		const char* SimpleModelVertexShaderSource =
#include "shaders/vertexshader.vert"
			;
		const char* SimpleModelFragmentShaderSource =
#include "shaders/fragmentshader.frag"
			;
		simpleModel = std::make_shared<Shader>(SimpleModelVertexShaderSource, SimpleModelFragmentShaderSource);
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
		case ShaderType::SimpleModel:
			return simpleModel;
		default:
			break;
		}
		return simpleModel;
	}
}
