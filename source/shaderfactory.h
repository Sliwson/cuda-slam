#pragma once

#include "shader.h"

namespace Common
{
    enum class ShaderType
    {
        simpleModel
    };

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
        Shader simpleModel;

    public:
        Shader& getShader(ShaderType type);
    };
}