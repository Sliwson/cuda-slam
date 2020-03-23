#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>

namespace Common
{
	// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
	enum class CameraMovement {
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		UP,
		DOWN
	};

	namespace CameraConsts
	{
		// Default camera values
		const float YAW = 180.0f;
		const float PITCH = -10.0f;
		const float SPEED = 5.0f;
		const float SENSITIVITY = 0.1f;
		const float FOV = 45.0f;
		const float NEAR_PLANE = 0.1f;
		const float FAR_PLANE = 100.0f;
	}

	// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
	class Camera
	{
	public:
		// Constructor with vectors
		Camera(glm::vec3 position, glm::vec3 cameraTarget, glm::vec3 worldUp);

		virtual ~Camera() {}

		void ChangePosition(glm::vec3 position);
		void ChangeTarget(glm::vec3 cameraTarget);
		void FollowObject(glm::vec3 targetPosition, glm::vec3 cameraToTarget);

		glm::vec3 GetPosition();
		glm::vec3 GetFront();
		float GetFov();;

		// Returns projection matrix
		glm::mat4 GetProjectionMatrix(float width, float height);

		// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
		glm::mat4 GetViewMatrix();

		// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
		virtual void ProcessKeyboard(CameraMovement direction, float deltaTime);

		// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
		virtual void ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch = true);

		// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
		void ProcessMouseScroll(float yoffset);
	protected:
		virtual void updateCameraVectors();

		// Camera Attributes
		glm::vec3 Position;
		glm::vec3 Front;
		glm::vec3 Up;
		glm::vec3 Right;
		glm::vec3 WorldUp;
		glm::vec3 CameraTarget;

		//camera options
		float NearPlane;
		float FarPlane;
		float Fov;
	};
}