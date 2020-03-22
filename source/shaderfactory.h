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

    private:
        ShaderFactory();

    public:
        ShaderFactory(ShaderFactory const&) = delete;
        void operator=(ShaderFactory const&) = delete;

    private:
        std::shared_ptr<Shader> simpleModel;

    public:
        std::shared_ptr<Shader> getShader(ShaderType type);
    };
}