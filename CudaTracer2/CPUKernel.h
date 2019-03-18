#ifndef H_CPUKERNEL
#define H_CPUKERNEL

#include <glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <memory>

struct Camera;

using namespace glm;
using namespace std;

namespace CPUKernel
{
	constexpr float EPSILON = 1e-3f;
	constexpr float INF = 3.402823466e+38F;

	enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC, MERGE };

	struct Material
	{
		Material(MaterialType type = DIFF, vec3 color = vec3(1), vec3 emission = vec3(0));
		MaterialType type;
		vec3 color;
		vec3 emission;
		float specular;
		float metalic;
		bool isTransparent = false;
		float nc = 1.0f;
		float nt = 1.5f;
	};

	struct Ray
	{
		vec3 origin;
		vec3 direction;
		vec3 invdir;
		int sign[3];

		Ray(vec3 origin, vec3 direction);
	};

	struct ObjectIntersection
	{
		ObjectIntersection(bool hit = false, float t = 0, vec3 normal = vec3(0), int materialID = 0, bool isLight = false);
		bool hit;
		float t;
		vec3 normal;
		bool islight;
		int materialID;
	};

	struct Sphere
	{
		Sphere(vec3 position = vec3(0), float radius = 0, int materialID = 0);
		float radius;
		vec3 position;
		int materialID;

		ObjectIntersection Intersect(const Ray &ray);
	};

	struct Light
	{
		virtual const vec3& GetPosition() = 0;
		virtual const vec3 GetRandomPointOnSurface() = 0;
		virtual const float& GetLuminance() = 0;
		virtual ObjectIntersection Intersect(const Ray &ray) = 0;
	};

	struct SphereLight : public Light
	{
		SphereLight(const vec3& position, const float& radius, const float& luminance);

		vec3 position;
		float radius;
		float luminance;

		virtual const vec3& GetPosition() override;
		virtual const vec3 GetRandomPointOnSurface() override;
		virtual const float& GetLuminance() override;
		virtual ObjectIntersection Intersect(const Ray &ray) override;
	};

	struct SunLight
	{
		float sunLuminance;
		float sunExtent;
		vec3 sunDirection;
	};

	struct RenderOption
	{
		unsigned int frame = 1;
		unsigned int maxSamples = 1;
		unsigned int loopX = 4, loopY = 4;
		unsigned int width, height;

		bool enableDof = false;
		bool isAccumulate;

		// SunLight
		SunLight sunLight;

		// Image Buffer
		float* data;
		float* previousFrameData;
	};

	struct KernelOption
	{
		unsigned int width, height;
		unsigned int xIndex, yIndex;
		unsigned int xUnit, yUnit;
		unsigned int frame;
		unsigned int maxDepth;
		bool enableDof;

		vector<Sphere> spheres;
		vector<Material> materials;
		vector<Light*> lights;

		float* data;
		float* previousFrameData;
	};

	void RenderKernel(const shared_ptr<Camera>& camera, const vector<Sphere>& spheres, const vector<Light*>& lights, RenderOption& option);

}
#endif
