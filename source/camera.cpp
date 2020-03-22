#include <glad/glad.h>
#include <glm/gtc/matrix_transform.hpp>
#include "camera.h"

namespace Common
{
    Camera::Camera() :
        Position(glm::vec3(0, 0, 0)), Front(glm::vec3(0, 0, 0)), Up(glm::vec3(0, 0, 0)), Right(glm::vec3(0, 0, 0)), WorldUp(glm::vec3(0, 0, 0)), CameraTarget(glm::vec3(0, 0, 0)),
        near_plane(CameraConsts::NEAR_PLANE), far_plane(CameraConsts::FAR_PLANE), fov(CameraConsts::FOV)
    {

    }

    Camera::Camera(glm::vec3 position, glm::vec3 cameraTarget, glm::vec3 worldUp) : Position(position), WorldUp(worldUp), CameraTarget(cameraTarget), near_plane(CameraConsts::NEAR_PLANE), far_plane(CameraConsts::FAR_PLANE), fov(CameraConsts::FOV)
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
		return fov;
	}

    // Returns projection matrix
    glm::mat4 Camera::GetProjectionMatrix(float width, float height)
    {
        return glm::perspective(glm::radians(fov), width / height, near_plane, far_plane);
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
        if (fov >= 1.0f && fov <= 90.0f)
            fov -= yoffset;
        if (fov <= 1.0f)
            fov = 1.0f;
        if (fov >= 90.0f)
            fov = 90.0f;
    }
}