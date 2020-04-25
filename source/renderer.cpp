#include <glm/gtc/matrix_transform.hpp>
#include <string>

#include "renderer.h"
#include "moveablecamera.h"
#include "shaderfactory.h"
#include "shadertype.h"
#include "Icosphere.h"

namespace Common
{
	std::vector<Renderer*> Renderer::renderers;

	namespace ShaderPath
	{
		const char* vertexShaderPath = "source/shaders/vertexshader.vert";
		const char* fragmentShaderPath = "source/shaders/fragmentshader.frag";
	}

	Renderer::Renderer(ShaderType shaderType, std::vector<Point_f> origin_points, std::vector<Point_f> result_points, std::vector<Point_f> cpu_points, std::vector<Point_f> gpu_points) :
		shaderType(shaderType), origin_points(origin_points), result_points(result_points), cpu_points(cpu_points), gpu_points(gpu_points)
	{
		width = 1280;
		height = 720;

		deltaTime = 0.0f;
		lastFrame = 0.0f;

		lastX = width / 2.0f;
		lastY = height / 2.0f;
		firstMouse = true;

		pointSize = 0.3f;
		pointScale = 1.0f;
		defaultScale = 10.0f;

		SetCamera(glm::vec3(1.5f * defaultScale, 0.0f, 0.0f));

		modelMatrix = glm::mat4(1.0f);
		normalMatrix = glm::mat3(glm::transpose(glm::inverse(modelMatrix)));

		renderers.push_back(this);

		Icosphere sphere(pointSize, 1, true);

		vertices = sphere.getInterleavedVerticesVector();
		indices = sphere.getIndicesVector();

		for (size_t i = 0; i < verticesVectorsCount; i++)
		{
			isVisible[i] = true;
		}

		SetModelMatrixToData();
	}

	Renderer::~Renderer()
	{
		renderers.erase(std::remove_if(renderers.begin(), renderers.end(), [&](Renderer* renderer) {return renderer == this; }));

		glDeleteVertexArrays(verticesVectorsCount, VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
		glDeleteBuffers(verticesVectorsCount, instanceVBO);
	}

	void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (IsRendererCreated(renderer) == false)
			return;

		glViewport(0, 0, width, height);

		renderer->width = width;
		renderer->height = height;
	}

	void Renderer::processInput(GLFWwindow* window)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (IsRendererCreated(renderer) == false)
			return;

		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::FORWARD, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::BACKWARD, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::LEFT, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::RIGHT, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::UP, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			renderer->camera->ProcessKeyboard(CameraMovement::DOWN, renderer->deltaTime);
		if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS)
		{
			renderer->pointSize += 0.01f;
		}
		if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
		{
			renderer->pointSize -= 0.01f;
			if (renderer->pointSize < 0.01f)
				renderer->pointSize = 0.01f;
		}
	}

	void Renderer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (IsRendererCreated(renderer) == false)
			return;

		if (action == GLFW_PRESS)
		{
			switch (key)
			{
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, true);
				break;
			case GLFW_KEY_1:
				renderer->isVisible[0] = !renderer->isVisible[0];
				break;
			case GLFW_KEY_2:
				renderer->isVisible[1] = !renderer->isVisible[1];
				break;
			case GLFW_KEY_3:
				renderer->isVisible[2] = !renderer->isVisible[2];
				break;
			case GLFW_KEY_4:
				renderer->isVisible[3] = !renderer->isVisible[3];
				break;
			default:
				break;
			}
		}
	}

	void Renderer::mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (IsRendererCreated(renderer) == false)
			return;

		if (renderer->firstMouse)
		{
			renderer->lastX = xpos;
			renderer->lastY = ypos;
			renderer->firstMouse = false;
		}

		float xoffset = xpos - renderer->lastX;
		float yoffset = renderer->lastY - ypos; // reversed since y-coordinates go from bottom to top

		renderer->lastX = xpos;
		renderer->lastY = ypos;

		renderer->camera->ProcessMouseMovement(xoffset, yoffset);
	}

	void Renderer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (IsRendererCreated(renderer) == false)
			return;

		renderer->camera->ProcessMouseScroll(yoffset);
	}

	void Renderer::Show()
	{
		InitWindow();
		SetShader();
		SetBuffers();
		MainLoop();
	}

	Renderer* Renderer::FindInstance(GLFWwindow* window)
	{
		for (Renderer* renderer : renderers)
		{
			if (renderer->window == window)
			{
				return renderer;
			}
		}
		return nullptr;
	}

	bool Renderer::IsRendererCreated(Renderer* renderer)
	{
		if (renderer == nullptr)
			return false;
		if (renderer->camera == nullptr)
			return false;
		return true;
	}

	int Renderer::InitWindow()
	{
		// glfw: initialize and configure
		// ------------------------------
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

		// glfw window creation
		// --------------------
		window = glfwCreateWindow(width, height, "Window Name Placeholder", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return -1;
		}

		glfwMakeContextCurrent(window);

		const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
		glfwSetWindowPos(window, mode->width / 2 - lastX, mode->height / 2 - lastY);

		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetKeyCallback(window, key_callback);
		glfwSetScrollCallback(window, scroll_callback);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		glfwSwapInterval(1);

		// glad: load all OpenGL function pointers
		// ---------------------------------------
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return -1;
		}

		glEnable(GL_DEPTH_TEST);

		return 0;
	}

	void Renderer::SetBuffers()
	{
		// create buffers/arrays
		glGenVertexArrays(verticesVectorsCount, VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);
		glGenBuffers(verticesVectorsCount, instanceVBO);

		//set data for VBO and EBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

		for (unsigned int i = 0; i < verticesVectorsCount; i++)
		{
			std::vector<Point_f>& vector = GetVector(i);
			glBindBuffer(GL_ARRAY_BUFFER, instanceVBO[i]);
			glBufferData(GL_ARRAY_BUFFER, vector.size() * sizeof(Point_f), &vector[0], GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			glBindVertexArray(VAO[i]);

			//bind vbo and ebo
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

			// set the vertex attribute pointers
			// vertex Positions
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
			// vertex normals
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
			//vertex offset using instancing
			glEnableVertexAttribArray(2);
			glBindBuffer(GL_ARRAY_BUFFER, instanceVBO[i]);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Point_f), (void*)0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glVertexAttribDivisor(2, 1);

			glBindVertexArray(0);
		}
	}

	void Renderer::MainLoop()
	{
		double previousTime = glfwGetTime();
		int frameCount = 0;
		std::string s = "";

		// render loop
		// -----------
		while (!glfwWindowShouldClose(window))
		{
			//per-frame time logic
			double currentFrame = glfwGetTime();
			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;
			frameCount++;
			if (currentFrame - previousTime >= 1.0)
			{
				s.append("Window Name Placeholder [FPS:");
				s.append(std::to_string((int)frameCount));
				s.append("]");
				glfwSetWindowTitle(window, s.c_str());
				s.clear();
				frameCount = 0;
				previousTime = currentFrame;
			}
			// input
			// -----
			processInput(window);

			// render
			// ------
			glClearColor(0.5f, 0.8f, 0.95f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			Draw();

			// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
			// -------------------------------------------------------------------------------
			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		// glfw: terminate, clearing all previously allocated GLFW resources.
		// ------------------------------------------------------------------
		glfwTerminate();
	}

	void Renderer::Draw()
	{
		shader->use();

		shader->setMat4("projection", camera->GetProjectionMatrix(width, height));
		shader->setMat4("view", camera->GetViewMatrix());
		shader->setMat4("model", modelMatrix);
		shader->setMat3("NormalMatrix", normalMatrix);

		shader->setFloat("PointSize", pointSize);
		shader->setFloat("PointScale", pointScale);

		shader->setVec3("viewPos", camera->GetPosition());

		shader->setVec3("lightDirection", glm::vec3(-0.2f, -1.0f, -0.3f));
		shader->setVec3("lightColor", glm::vec3(1.0f, 1.0f, 1.0f));

		for (unsigned int i = 0; i < verticesVectorsCount; i++)
		{
			if (isVisible[i] == true)
			{
				std::vector<Point_f>& vector = GetVector(i);

				shader->setVec3("objectColor", GetColor(i));

				// uncomment to enable rotation
				//glm::mat4 tmp = modelMatrix;

				////modelMatrix = glm::translate(modelMatrix, glm::vec3(i, i, i));
				//modelMatrix = glm::rotate(modelMatrix, (float)i, glm::vec3(0.5f, 0.5f, 0.5f));

				//SetModelMatrix(modelMatrix);
				//shader->setMat4("model", modelMatrix);
				//shader->setMat3("NormalMatrix", normalMatrix);

				//SetModelMatrix(tmp);

				glBindVertexArray(VAO[i]);
				glDrawElementsInstanced(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0, vector.size());
				glBindVertexArray(0);
			}
		}
	}

	std::vector<Point_f>& Renderer::GetVector(int index)
	{
		switch (index)
		{
		case 0:
			return origin_points;
		case 1:
			return result_points;
		case 2:
			return cpu_points;
		case 3:
			return gpu_points;
		default:
			return origin_points;
		}
	}

	glm::vec3 Renderer::GetColor(int index)
	{
		switch (index)
		{
		case 0://origin
			return glm::vec3(0.8f, 0.8f, 0.8f);
		case 1://result
			return glm::vec3(0.0f, 0.0f, 1.0f);
		case 2://cpu
			return glm::vec3(1.0f, 0.0f, 0.0f);
		case 3://gpu
			return glm::vec3(0.0f, 1.0f, 0.0f);
		default:
			return glm::vec3(0.8f, 0.8f, 0.8f);
		}
	}

	void Renderer::SetModelMatrixToData()
	{
		if (result_points.size() == 0)
			return;

		float max[3];
		float min[3];

		max[0] = result_points[0].x;
		max[1] = result_points[0].y;
		max[2] = result_points[0].z;
		min[0] = result_points[0].x;
		min[1] = result_points[0].y;
		min[2] = result_points[0].z;

		for (unsigned int i = 1; i < result_points.size(); i++)
		{
			if (result_points[i].x > max[0])
				max[0] = result_points[i].x;
			if (result_points[i].y > max[1])
				max[1] = result_points[i].y;
			if (result_points[i].z > max[2])
				max[2] = result_points[i].z;
			if (result_points[i].x < min[0])
				min[0] = result_points[i].x;
			if (result_points[i].y < min[1])
				min[1] = result_points[i].y;
			if (result_points[i].z < min[2])
				min[2] = result_points[i].z;
		}

		float middle[3];
		float max_range = 0.0f;
		for (int i = 0; i < 3; i++)
		{
			middle[i] = (max[i] + min[i]) / 2.0f;
			if (middle[i] > max_range)
				max_range = middle[i];
		}

		if (max_range < 1e-4)
			max_range = defaultScale / 2.0f;

		pointScale = defaultScale / (2.0f * max_range);

		glm::vec3 middleVector = glm::vec3(middle[0], middle[1], middle[2]);

		modelMatrix = glm::scale(modelMatrix, glm::vec3(pointScale));
		modelMatrix = glm::translate(modelMatrix, -middleVector);

		SetModelMatrix(modelMatrix);
	}

	void Renderer::SetShader()
	{
		ShaderFactory& sf = ShaderFactory::getInstance();
		shader = sf.getShader(shaderType);
	}

	void Renderer::SetCamera(glm::vec3 position)
	{
		camera = std::make_unique<MoveableCamera>(position);
	}

	glm::mat4 Renderer::GetModelMatrix()
	{
		return modelMatrix;
	}

	void Renderer::SetModelMatrix(glm::mat4 model)
	{
		modelMatrix = model;
		normalMatrix = glm::mat3(glm::transpose(glm::inverse(model)));
	}
}