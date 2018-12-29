#ifndef H_PATHKERNEL
#define H_PATHKERNEL

#include "TracingCommon.h"
#include "KDTree.h"

#include <windows.h> 
#include <iostream>
#include <memory>

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr float EPSILON = 1e-3f;
constexpr float INF = 3.402823466e+38F;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

struct Ray
{
	vec3 origin;
	vec3 direction;
	__host__ __device__ Ray(vec3 origin, vec3 direction)
	{
		this->origin = origin;
		this->direction = direction;
	}
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

struct Sphere
{
	__host__ __device__ Sphere(vec3 position = vec3(0), float radius = 0, int materialID = 0)
	{
		this->position = position;
		this->radius = radius;
		this->materialID = materialID;
	}
	float radius;
	vec3 position;
	int materialID;
	__device__ ObjectIntersection Intersect(const Ray &ray)
	{
		bool hit = false;
		float distance = 0, t = 0;
		vec3 normal = vec3(0, 0, 0);
		vec3 op = position - ray.origin;
		float b = dot(op, ray.direction);
		float det = b * b - dot(op, op) + radius * radius;

		if (det < EPSILON)
			return ObjectIntersection(hit, t, normal, materialID);
		else
			det = glm::sqrt(det);

		distance = (t = b - det) > EPSILON ? t : ((t = b + det) > EPSILON ? t : 0);
		if (distance > EPSILON)
		{
			hit = true;
			normal = normalize(ray.direction * distance - op);
		}
		ObjectIntersection result = ObjectIntersection(hit, distance, normal, materialID);
		return result;
	}
};

template <typename T>
struct KernelArray
{
	T*  array;
	int size;
};

template <typename T>
KernelArray<T> ConvertToKernel(thrust::device_vector<T>& dVec)
{
	KernelArray<T> kArray;
	kArray.array = thrust::raw_pointer_cast(&dVec[0]);
	kArray.size = (int) dVec.size();

	return kArray;
}

struct KernelOption
{
	int frame;
	int loopX, loopY;
	unsigned long seed;
	bool enableDof;
	int maxSamples;
	int maxDepth;
	int hdrHeight, hdrWidth;

	// SunLight
	vec3 sunDirection;
	float sunLuminance;
	float sunExtent;

	// Test Material
	float specular;
	float metalic;


	KernelArray<Sphere> spheres;
	
	ivec3* vertexIndices;
	ivec3* normalIndices;
	int* materialIndices;
	vec3* verts;
	vec3* norms;
	Material* materials;
	int kdTreeRootIndex;
	KDTreeNode* kdTreeNodes;
	int* kdTreeTriIndices;
	cudaSurfaceObject_t surface;
};

struct RenderOption
{
	int frame = 1;
	int maxSamples = 1;
	int loopX = 4, loopY = 4;
	bool enableDof = false;
	bool isAccumulate;

	// Test Material
	float specular;
	float metalic;

	// SunLight
	float sunLuminance;
	float sunExtent;
	vec3 sunDirection;

	cudaSurfaceObject_t surf;
};

void InitHDRTexture(const char* hdrFileName);

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, KDTree* tree, const RenderOption& option);

#endif