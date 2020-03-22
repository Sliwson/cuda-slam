#include "common.h"
#include "moveablecamera.h"
#include "shaderfactory.h"
#include "shadertype.h"



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
		SetCamera(glm::vec3(4.0f, 0.0f, 0.0f));

		width = 1280;
		height = 720;

		deltaTime = 0.0f;
		lastFrame = 0.0f;

		lastX = width / 2.0f;
		lastY = height / 2.0f;
		firstMouse = true;

		pointSize = 1.0f;

		modelMatrix = glm::mat4(1.0f);

		renderers.push_back(this);
	}

	Renderer::~Renderer()
	{
		std::vector<Renderer*>::iterator it;
		for (it = renderers.begin(); it != renderers.end(); it++)
		{
			if ((*it) == this)
			{
				renderers.erase(it);
				break;
			}
		}
		for (unsigned int i = 0; i < 4; i++)
		{
			glDeleteVertexArrays(1, &VAO[i]);
			glDeleteBuffers(1, &VBO[i]);
		}
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

	void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (renderer == nullptr)
			return;
		if (renderer->camera == nullptr)
			return;

		glViewport(0, 0, width, height);

		renderer->width = width;
		renderer->height = height;
	}

	void Renderer::processInput(GLFWwindow* window)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (renderer == nullptr)
			return;
		if (renderer->camera == nullptr)
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
			renderer->pointSize += 0.1f;
		if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS)
		{
			renderer->pointSize -= 0.1f;
			if (renderer->pointSize < 0.0f)
				renderer->pointSize = 0.0f;
		}
	}

	void Renderer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (renderer == nullptr)
			return;

		if (action == GLFW_PRESS)
		{
			switch (key)
			{
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, true);
				break;
			default:
				break;
			}
		}
	}

	void Renderer::mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		Renderer* renderer = Renderer::FindInstance(window);
		if (renderer == nullptr)
			return;
		if (renderer->camera == nullptr)
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
		if (renderer == nullptr)
			return;
		if (renderer->camera == nullptr)
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
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

		return 0;
	}

	void Renderer::SetBuffers()
	{
		for (unsigned int i = 0; i < 4; i++)
		{
			std::vector<Point_f>& vertices = GetVector(i);
			// create buffers/arrays
			glGenVertexArrays(1, &VAO[i]);
			glGenBuffers(1, &VBO[i]);

			glBindVertexArray(VAO[i]);
			// load data into vertex buffers
			glBindBuffer(GL_ARRAY_BUFFER, VBO[i]);

			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Point_f), &vertices[0], GL_STATIC_DRAW);

			// set the vertex attribute pointers
			// vertex Positions
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Point_f), (void*)0);

			glBindVertexArray(0);
		}
	}

	void Renderer::MainLoop()
	{
		// render loop
		// -----------
		while (!glfwWindowShouldClose(window))
		{
			//per-frame time logic
			double currentFrame = glfwGetTime();
			deltaTime = currentFrame - lastFrame;
			lastFrame = currentFrame;
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

		shader->setVec3("viewPos", camera->GetPosition());

		shader->setFloat("pointRadius", pointSize);
		//shader->setFloat("pointScale", height / tanf(glm::radians(camera->GetFov() * 0.5f)));

		for (unsigned int i = 0; i < 4; i++)
		{
			shader->setVec3("color", GetColor(i));
			std::vector<Point_f>& vertices = GetVector(i);
			glBindVertexArray(VAO[i]);
			glDrawArrays(GL_POINTS, 0, vertices.size());
			glBindVertexArray(0);
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
			origin_points;
		}
	}

	glm::vec3 Renderer::GetColor(int index)
	{
		switch (index)
		{
		case 0://origin
			return glm::vec3(0.0f, 0.0f, 0.0f);
		case 1://result
			return glm::vec3(0.0f, 0.0f, 1.0f);
		case 2://cpu
			return glm::vec3(1.0f, 0.0f, 0.0f);
		case 3://gpu
			return glm::vec3(0.0f, 1.0f, 0.0f);
		default:
			return glm::vec3(0.0f, 0.0f, 0.0f);
		}
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
	
	

}