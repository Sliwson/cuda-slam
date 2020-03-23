#pragma once

#include <memory>

#include "shader.h"

namespace Common
{
	enum class ShaderType;

	class ShaderFactory
	{
	public:
		static ShaderFactory& getInstance();

		std::shared_ptr<Shader> getShader(ShaderType type);

		ShaderFactory(ShaderFactory const&) = delete;
		void operator=(ShaderFactory const&) = delete;

	private:
		ShaderFactory();

		std::shared_ptr<Shader> simpleModel;
	};
}