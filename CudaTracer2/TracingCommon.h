#ifndef TRACINGCOMMON_H
#define TRACINGCOMMON_H

#include <iostream>
#include <memory>

#include <GL/glew.h>
#include <GL/glfw3.h>

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda.h>
#include <curand_kernel.h>
#include <windows.h> 
#include <cuda_gl_interop.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

using namespace glm;
using namespace std;

constexpr float EPSILON = 1e-3f;

enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC };

struct Ray
{
	vec3 origin;
	vec3 direction;
	__host__ __device__ Ray(vec3 origin, vec3 direction)
	{
		this->origin = origin + direction;
		this->direction = direction;
	}
};

struct Material
{
	__host__ __device__ Material(MaterialType type = DIFF, vec3 color = vec3(0), vec3 emission = vec3(0))
	{
		this->type = type;
		this->color = color;
		this->emission = emission;
	}
	MaterialType type;
	vec3 color;
	vec3 emission;
};

struct ObjectIntersection
{
	__host__ __device__ ObjectIntersection(bool hit = false, float t = 0, vec3 normal = vec3(0), int materialID = 0)
	{
		this->hit = hit;
		this->t = t;
		this->normal = normal;
		this->materialID = materialID;
	}
	bool hit;
	float t;
	vec3 normal;
	int materialID;
};

struct Triangle
{
	__host__ __device__ Triangle(vec3 pos0 = vec3(0), vec3 pos1 = vec3(0), vec3 pos2 = vec3(0), vec3 nor0 = vec3(0), vec3 nor1 = vec3(0), vec3 nor2 = vec3(0), int materialID = 0)
	{
		pos[0] = pos0;
		pos[1] = pos1;
		pos[2] = pos2;
		nor[0] = normalize(nor0);
		nor[1] = normalize(nor1);
		nor[2] = normalize(nor2);
		this->materialID = materialID;
	}

	vec3 pos[3];
	vec3 nor[3];
	int materialID;
};

struct CameraOption
{
	float width, height;
	vec3 position;
	float moveSpeed, mouseSpeed;
	float nearPlane, farPlane;
	float fov;
	float pitch, yaw;
	float aperture, focalDistance;
};

struct Camera
{
	__host__ __device__ Camera(CameraOption option)
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
	}

	__device__ Ray GetRay(curandState* randState, int x, int y, bool enableDof)
	{
		float jitterValueX = curand_uniform(randState) - 0.5;
		float jitterValueY = curand_uniform(randState) - 0.5;

		vec3 wDir = glm::normalize(-forward);
		vec3 uDir = glm::normalize(cross(up, wDir));
		vec3 vDir = glm::cross(wDir, -uDir);

		float top = tan(fov * glm::pi<float>() / 360.0f);
		float right = aspectRatio * top;
		float bottom = -top;
		float left = -right;

		float imPlaneUPos = left + (right - left)*(((float) x + jitterValueX) / (float) width);
		float imPlaneVPos = bottom + (top - bottom)*(((float) y + jitterValueY) / (float) height);

		vec3 originDirection = imPlaneUPos * uDir + imPlaneVPos * vDir - wDir;
		vec3 pointOnImagePlane = position + ((originDirection) * focalDistance);
		if (enableDof)
		{
			vec3 aperturePoint = vec3(0, 0, 0);

			if (aperture >= EPSILON)
			{
				float r1 = curand_uniform(randState);
				float r2 = curand_uniform(randState);

				float angle = two_pi<float>() * r1;
				float distance = aperture * sqrt(r2);
				float apertureX = cos(angle) * distance;
				float apertureY = sin(angle) * distance;

				aperturePoint = position + (wDir * apertureX) + (uDir * apertureY);
			}
			else
			{
				aperturePoint = position;
			}
			return Ray(aperturePoint, normalize(pointOnImagePlane - aperturePoint));
		}
		else
		{
			return Ray(position, normalize(originDirection));
		}
	}

	void UpdateScreen(int width, int height)
	{

	}

	void UpdateCamera(float deltaTime)
	{
		this->width = width;
		this->height = height;
		this->aspectRatio = width / (float) height;

		glViewport(0, 0, width, height);
		proj = perspective(radians(fov), aspectRatio, nearPlane, farPlane);
	}

	bool toggleMouseMovement;
	float width, height;
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