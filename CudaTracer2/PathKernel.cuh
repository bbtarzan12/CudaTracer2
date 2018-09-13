#ifndef H_PATHKERNEL
#define H_PATHKERNEL

#include "TracingCommon.h"

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

#include "device_launch_parameters.h"

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

constexpr float EPSILON = 1e-3f;

enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC };

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

struct Material
{
	__host__ __device__ Material(MaterialType type = DIFF, vec3 color = vec3(1), vec3 emission = vec3(0))
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
};

struct RenderOption
{
	int frame = 1;
	int maxSamples = 1;
	int loopX = 1, loopY = 1;
	bool enableDof = false;
	bool isAccumulate;
	cudaSurfaceObject_t surf;
};

__host__ __device__ unsigned int WangHash(unsigned int a);

__device__ ObjectIntersection Intersect(Ray ray, KernelArray<Sphere> spheres);

__device__ Ray GetReflectedRay(Ray ray, vec3 hitPoint, glm::vec3 normal, vec3 &mask, Material material, curandState* randState);

__device__ vec3 TraceRay(Ray ray, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, curandState* randState);

__global__ void PathImageKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, cudaSurfaceObject_t surface);

__global__ void PathAccumulateKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, cudaSurfaceObject_t surface);

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, const thrust::host_vector<Material>& materials, const RenderOption& option);

#endif