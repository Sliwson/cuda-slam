#pragma once

#include <memory>

#include "shader.h"

namespace Common
{
	enum class ShaderType;

	class ShaderFactory
	{
	public:
		ShaderFactory();
		ShaderFactory(ShaderFactory const&) = delete;
		void operator=(ShaderFactory const&) = delete;
		
		std::shared_ptr<Shader> GetShader(ShaderType type);

	private:

		std::shared_ptr<Shader> simpleModel;
	};
}
