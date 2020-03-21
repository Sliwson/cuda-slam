#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <vector>

#include "shader.h"
#include "camera.h"

namespace Common
{
	namespace ShaderPath
	{
		const char* vertexShaderPath = "source/shaders/vertexshader.vert";
		const char* fragmentShaderPath = "source/shaders/fragmentshader.frag";
	}


	class Renderer
	{
	public:
		Renderer();

		~Renderer();


		//callbacks
		static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
		static void processInput(GLFWwindow* window);
		static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
		static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


	private:
		

		int InitWindow();
		void SetShader();

		static std::vector<GLFWwindow*> active_windows;

		Shader shader;
		Camera* camera;

		GLFWwindow* window;

		//window size
		int width;
		int height;

	};
}