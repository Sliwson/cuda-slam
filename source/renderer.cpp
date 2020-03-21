#include "renderer.h"
#include "shaderfactory.h"
#include "moveablecamera.h"


namespace Common
{
    Renderer::Renderer() :
        shader(ShaderFactory::getInstance().getShader(ShaderType::simpleModel))
    {
        SetCamera(glm::vec3(4.0f, 2.0f, 0.0f));

        width = 1280;
        height = 720;

        deltaTime = 0.0f;
        lastFrame = 0.0f;

        lastX = width / 2.0f;
        lastY = height / 2.0f;
        firstMouse = true;
    }

    Renderer::~Renderer()
    {
        delete camera;
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
    }

    void Renderer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
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


	int Renderer::InitWindow()
	{
        // glfw: initialize and configure
        // ------------------------------
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

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

            // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
            // -------------------------------------------------------------------------------
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        // glfw: terminate, clearing all previously allocated GLFW resources.
        // ------------------------------------------------------------------
        glfwTerminate();

        return 0;
	}


	void Renderer::SetShader(ShaderType type)
	{
        ShaderFactory& sf = ShaderFactory::getInstance();
        shader = sf.getShader(type);
	}

    void Renderer::SetCamera(glm::vec3 position)
    {
        camera = new MoveableCamera(position);
    }
    
    

}