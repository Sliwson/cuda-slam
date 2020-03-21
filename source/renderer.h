#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <vector>

#include "shader.h"

namespace Common
{
	class Camera;
	enum class ShaderType;

	class Renderer
	{
	public:
		Renderer();

		~Renderer();




		static Renderer* FindInstance(GLFWwindow* window);
		//callbacks
		static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
		static void processInput(GLFWwindow* window);
		static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
		static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


		int InitWindow();
	private:
		

		
		void SetShader(ShaderType type);
		void SetCamera(glm::vec3 position);

		static std::vector<Renderer*> renderers;

		Shader shader;
		Camera* camera;

		GLFWwindow* window;

		//window size
		int width;
		int height;

		//frames
		double deltaTime;
		double lastFrame;

		//mouse
		float lastX;
		float lastY;
		bool firstMouse;

	};
}