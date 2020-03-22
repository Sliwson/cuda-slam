#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <vector>
#include <memory>

#include "shader.h"

namespace Common
{
	class Camera;
	enum class ShaderType;

	using Point_f = Point<float>;

	class Renderer
	{
	public:
		Renderer(ShaderType shaderType, std::vector<Point_f> origin_points, std::vector<Point_f> result_points, std::vector<Point_f> cpu_points, std::vector<Point_f> gpu_points);

		~Renderer();


		static Renderer* FindInstance(GLFWwindow* window);
		//callbacks
		static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
		static void processInput(GLFWwindow* window);
		static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
		static void mouse_callback(GLFWwindow* window, double xpos, double ypos);
		static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

		void Show();
		
	private:
		int InitWindow();
		void SetBuffers();

		void MainLoop();
		void Draw();

		std::vector<Point_f>& GetVector(int index);
		glm::vec3 GetColor(int index);

		
		
		void SetShader();
		void SetCamera(glm::vec3 position);

		static std::vector<Renderer*> renderers;

		std::shared_ptr<Shader> shader;
		std::unique_ptr<Camera> camera;

		GLFWwindow* window;

		//shader
		ShaderType shaderType;

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

		//point size
		float pointSize;

		//data
		std::vector<Point_f> origin_points;
		std::vector<Point_f> result_points;
		std::vector<Point_f> cpu_points;
		std::vector<Point_f> gpu_points;

		//render data
		unsigned int VAO[4];
		unsigned int VBO[4];

		//model matrix
		glm::mat4 modelMatrix;
	};
}