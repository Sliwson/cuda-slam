#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"

namespace Common
{
	Camera::Camera(glm::vec3 position, glm::vec3 cameraTarget, glm::vec3 worldUp) :
		Position(position), WorldUp(worldUp), CameraTarget(cameraTarget), NearPlane(CameraConsts::NEAR_PLANE), FarPlane(CameraConsts::FAR_PLANE), Fov(CameraConsts::FOV)
	{
		updateCameraVectors();
	}

	void Camera::ChangePosition(glm::vec3 position)
	{
		Position = position;
		updateCameraVectors();
	}

	void Camera::ChangeTarget(glm::vec3 cameraTarget)
	{
		CameraTarget = cameraTarget;
		updateCameraVectors();
	}

	void Camera::FollowObject(glm::vec3 targetPosition, glm::vec3 cameraToTarget)
	{
		CameraTarget = targetPosition;
		Position = targetPosition - cameraToTarget;
		updateCameraVectors();
	}

	glm::vec3 Camera::GetPosition()
	{
		return Position;
	}

	glm::vec3 Camera::GetFront()
	{
		return Front;
	}

	float Camera::GetFov()
	{
		return Fov;
	}

	// Returns projection matrix
	glm::mat4 Camera::GetProjectionMatrix(float width, float height)
	{
		return glm::perspective(glm::radians(Fov), width / height, NearPlane, FarPlane);
	}

	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	glm::mat4 Camera::GetViewMatrix()
	{
		return glm::lookAt(Position, Position + Front, Up);
	}

	void Camera::updateCameraVectors()
	{
		Front = glm::normalize(CameraTarget - Position);
		Right = glm::normalize(glm::cross(WorldUp, -Front));
		Up = glm::cross(-Front, Right);
	}

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void Camera::ProcessKeyboard(CameraMovement direction, float deltaTime)
	{
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch)
	{
	}

	// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
	void Camera::ProcessMouseScroll(float yoffset)
	{
		if (Fov >= 1.0f && Fov <= 90.0f)
			Fov -= yoffset;
		if (Fov <= 1.0f)
			Fov = 1.0f;
		if (Fov >= 90.0f)
			Fov = 90.0f;
	}
}
