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

	std::shared_ptr<Shader> ShaderFactory::GetShader(ShaderType type)
	{
		switch (type)
		{
		case ShaderType::SimpleModel:
			return simpleModel;
		default:
			return simpleModel;
		}
	}
}
