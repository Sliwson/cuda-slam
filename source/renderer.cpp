

#include "renderer.h"
#include "shaderfactory.h"


namespace Common
{


    void Renderer::framebuffer_size_callback(GLFWwindow* window, int width, int height)
    {

    }

    void Renderer::processInput(GLFWwindow* window)
    {
    }

    void Renderer::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
    }

    void Renderer::mouse_callback(GLFWwindow* window, double xpos, double ypos)
    {
    }

    void Renderer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
    {
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
        GLFWwindow* window = glfwCreateWindow(width, height, "Window Name Placeholder", NULL, NULL);
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
	}

	void Renderer::SetShader()
	{
        ShaderFactory& sf = ShaderFactory::getInstance();
        shader = sf.getShader(ShaderType::simpleModel);
	}
    
}