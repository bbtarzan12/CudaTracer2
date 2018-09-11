#ifndef TRACINGCOMMON_H
#define TRACINGCOMMON_H

#include <windows.h> 
#include <iostream>
#include <memory>

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace glm;
using namespace std;

struct CameraOption
{
	int width, height;
	vec3 position;
	float moveSpeed, mouseSpeed;
	float nearPlane, farPlane;
	float fov;
	float pitch, yaw;
	float aperture, focalDistance;
};

struct Camera
{
	Camera(CameraOption option)
	{
		width = option.width;
		height = option.height;
		moveSpeed = option.mouseSpeed;
		mouseSpeed = option.mouseSpeed;
		nearPlane = option.nearPlane;
		farPlane = option.farPlane;
		fov = option.fov;
		pitch = option.pitch;
		yaw = option.yaw;
		aperture = option.aperture;
		focalDistance = option.focalDistance;
		position = option.position;
	}

	void UpdateScreen(int width, int height)
	{
		this->width = width;
		this->height = height;
		this->aspectRatio = width / (float) height;
	}

	void UpdateCamera(float deltaTime, vec2 keyboardInput = vec2(0), vec2 mouseInput = vec2(0))
	{
		pitch += mouseSpeed * mouseInput.x * deltaTime;
		yaw += mouseSpeed * mouseInput.y * deltaTime;

		pitch = clamp(pitch, -89.0f, 89.0f);
		yaw = mod(yaw, 360.0f);

		position += (forward * keyboardInput.x + right * keyboardInput.y) * moveSpeed * deltaTime;
		forward.x = cos(radians(pitch)) * sin(radians(yaw));
		forward.y = sin(radians(pitch));
		forward.z = cos(radians(pitch)) * cos(radians(yaw));
		forward = normalize(forward);
		right = normalize(cross(forward, vec3(0, 1, 0)));
		up = normalize(cross(right, forward));

		view = lookAt(position, position + forward, up);
		proj = perspective(radians(fov), aspectRatio, nearPlane, farPlane);

		if (length(keyboardInput) != 0 || length(mouseInput) != 0)
			dirty = true;
	}

	void ResetDirty()
	{
		dirty = false;
	}

	bool dirty = true;
	int width, height;
	float moveSpeed, mouseSpeed;
	float nearPlane, farPlane;
	float fov;
	float aspectRatio;
	float pitch, yaw;

	// fov
	float aperture, focalDistance;

	vec3 position;
	vec3 forward, up, right;

	mat4 view;
	mat4 proj;
};

#endif