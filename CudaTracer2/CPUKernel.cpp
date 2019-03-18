#include "CPUKernel.h"
#include "TracingCommon.h"
#include <random>
#include <iostream>
#include <time.h>

using namespace CPUKernel;

float* previousBuffer;

uint32_t seed = 1;
uint32_t random()
{

	uint32_t t = time(nullptr);
	seed ^= t << 15;
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

float fRandom()
{
	return (float) random() / 0xffffffff;
}

Material::Material(MaterialType type /*= DIFF*/, vec3 color /*= vec3(1)*/, vec3 emission /*= vec3(0)*/)
{
	this->type = type;
	this->color = color;
	this->emission = emission;
}


Ray::Ray(vec3 origin, vec3 direction)
{
	this->origin = origin;
	this->direction = direction;
	invdir = 1.0f / direction;
	sign[0] = (invdir.x < 0);
	sign[1] = (invdir.y < 0);
	sign[2] = (invdir.z < 0);
}

ObjectIntersection::ObjectIntersection(bool hit /*= false*/, float t /*= 0*/, vec3 normal /*= vec3(0)*/, int materialID /*= 0*/, bool isLight)
{
	this->hit = hit;
	this->t = t;
	this->normal = normal;
	this->materialID = materialID;
	this->islight = isLight;
}

Sphere::Sphere(vec3 position /*= vec3(0)*/, float radius /*= 0*/, int materialID /*= 0*/)
{
	this->position = position;
	this->radius = radius;
	this->materialID = materialID;
}

ObjectIntersection Sphere::Intersect(const Ray &ray)
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

ObjectIntersection Intersect(Ray ray, vector<Sphere>& spheres, vector<Light*>& lights)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp = ObjectIntersection();

	intersection.t = INF;

	for (auto & sphere : spheres)
	{
		temp = sphere.Intersect(ray);

		if (temp.hit)
		{
			if (intersection.t == 0 || temp.t < intersection.t)
			{
				intersection = temp;
			}
		}
	}

	for (auto & light : lights)
	{
		temp = light->Intersect(ray);

		if (temp.hit)
		{
			if (intersection.t == 0 || temp.t < intersection.t)
			{
				intersection = temp;
			}
		}
	}

	return intersection;
}

Ray GetReflectedRay(Ray ray, vec3 hitPoint, vec3 normal, vec3 &mask, Material material)
{
	switch (material.type)
	{
		case DIFF:
		{
			vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
			float phi = two_pi<float>() * fRandom();
			float r2 = fRandom();
			float r2s = sqrt(r2);

			vec3 w = nl;
			vec3 u;
			if (fabs(w.x) > 0.1f)
				u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
			else
				u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
			vec3 v = cross(w, u);
			vec3 reflected = normalize((u * cosf(phi) * r2s + v * sinf(phi) * r2s + w * sqrt(1 - r2)));

			mask *= material.color;
			return Ray(hitPoint, reflected);
		}
		case GLOSS:
		{
			float phi = 2 * pi<float>() * fRandom();
			float r2 = fRandom();
			float cosTheta = powf(1 - r2, 1.0f / (20 + 1));
			float sinTheta = sqrt(1 - cosTheta * cosTheta);

			vec3 w = normalize(ray.direction - normal * 2.0f * dot(normal, ray.direction));
			vec3 u = normalize(cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
			vec3 v = cross(w, u);

			vec3 reflected = normalize(u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta);
			mask *= material.color;
			return Ray(hitPoint, reflected);
		}
		case TRANS:
		{
			vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
			vec3 reflection = ray.direction - normal * 2.0f * dot(normal, ray.direction);
			bool into = dot(normal, nl) > EPSILON;

			float nc = 1.0f;
			float nt = 1.5f;

			float nnt = into ? nc / nt : nt / nc;

			float Re, RP, TP, Tr;
			vec3 tdir = vec3(0.0f, 0.0f, 0.0f);

			float ddn = dot(ray.direction, nl);
			float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);

			if (cos2t < EPSILON) return Ray(hitPoint, reflection);

			if (into)
				tdir = normalize((ray.direction * nnt - normal * (ddn * nnt + sqrt(cos2t))));
			else
				tdir = normalize((ray.direction * nnt + normal * (ddn * nnt + sqrt(cos2t))));

			float a = nt - nc;
			float b = nt + nc;
			float R0 = a * a / (b * b);

			float c;
			if (into)
				c = 1 + ddn;
			else
				c = 1 - dot(tdir, normal);

			Re = R0 + (1 - R0) * c * c * c * c * c;
			Tr = 1 - Re;

			float P = .25 + .5 * Re;
			RP = Re / P;
			TP = Tr / (1 - P);

			if (fRandom() < P)
			{
				mask *= (RP);
				return Ray(hitPoint, reflection);
			}
			mask *= (TP);
			return Ray(hitPoint, tdir);
		}
		case SPEC:
		{
			vec3 reflected = ray.direction - normal * 2.0f * dot(normal, ray.direction);
			mask *= material.color;
			return Ray(hitPoint, reflected);
		}
	}
}

vec3 TraceRay(Ray ray, KernelOption option)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < option.maxDepth; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, option.spheres, option.lights);

		if (!intersection.hit)
		{
			return resultColor;
		}

		vec3& hitPoint = ray.origin + ray.direction * intersection.t;
		Material& hitMaterial = option.materials[intersection.materialID];
		vec3& emission = hitMaterial.emission;

		//if (depth == 0)
		//{
		//	for (auto & light : option.lights)
		//	{
		//		vec3 lightRandomPoint = light->GetRandomPointOnSurface();
		//		vec3 lightPosition = light->GetPosition();
		//		vec3 lightNormal = normalize(light->GetPosition() - hitPoint);

		//		Ray lightRay = Ray(hitPoint + lightNormal * EPSILON, lightNormal);
		//		ObjectIntersection lightIntersection = Intersect(lightRay, option.spheres, option.lights);

		//		if (lightIntersection.islight)
		//		{
		//			Material& lightMaterial = option.materials[lightIntersection.materialID];
		//			resultColor += mask * lightMaterial.emission;
		//		}
		//	}
		//}

		float maxReflection = max(max(mask.r, mask.g), mask.b);
		if (fRandom() > maxReflection)
			break;

		resultColor += mask * emission;
		ray = GetReflectedRay(ray, hitPoint, intersection.normal, mask, hitMaterial);
		mask *= 1 / maxReflection;
	}
	return resultColor;
}

Ray GetRay(Camera* camera, int x, int y, bool enableDof)
{
	float jitterValueX = 2 * fRandom() - 1.0f;
	float jitterValueY = 2 * fRandom() - 1.0f;

	vec3 wDir = normalize(-camera->forward);
	vec3 uDir = normalize(cross(camera->up, wDir));
	vec3 vDir = cross(wDir, -uDir);

	float top = tan(camera->fov * pi<float>() / 360.0f);
	float right = camera->aspectRatio * top;
	float bottom = -top;
	float left = -right;

	float imPlaneUPos = left + (right - left)*(((float) x + jitterValueX) / (float) camera->width);
	float imPlaneVPos = bottom + (top - bottom)*(((float) y + jitterValueY) / (float) camera->height);

	vec3 originDirection = imPlaneUPos * uDir + imPlaneVPos * vDir - wDir;
	vec3 pointOnImagePlane = camera->position + ((originDirection) * camera->focalDistance);
	if (enableDof)
	{
		vec3 aperturePoint = vec3(0, 0, 0);

		if (camera->aperture >= EPSILON)
		{
			float r1 = fRandom();
			float r2 = fRandom();

			float angle = two_pi<float>() * r1;
			float distance = camera->aperture * sqrt(r2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = camera->position + (wDir * apertureX) + (uDir * apertureY);
		}
		else
		{
			aperturePoint = camera->position;
		}
		return Ray(aperturePoint, normalize(pointOnImagePlane - aperturePoint));
	}
	else
	{
		return Ray(camera->position, normalize(originDirection));
	}
}


void PathAccumulateKernel(Camera* camera, KernelOption option)
{

#pragma omp parallel for
	for (int i = 0; i < option.xUnit * option.yUnit; i++)
	{
		unsigned int x = (i / option.yUnit) + option.xUnit * option.xIndex;
		unsigned int y = (i % option.yUnit) + option.yUnit * option.yIndex;
		unsigned int index = (x + y * option.width) * 4;


		if (index >= option.width * option.height * 4)
			continue;

		vec3 originalColor = vec3(option.previousFrameData[index], option.previousFrameData[index + 1], option.previousFrameData[index + 2]);
		vec3 resultColor = vec3(0, 0, 0);
		Ray ray = GetRay(camera, x, y, option.enableDof);
		vec3 color = TraceRay(ray, option);
		resultColor = (vec3(originalColor.x, originalColor.y, originalColor.z) * (float) (option.frame - 1) + color) / (float) option.frame;
		option.data[index + 0] = resultColor.r;
		option.data[index + 1] = resultColor.g;
		option.data[index + 2] = resultColor.b;
		option.data[index + 3] = 1.0;
	}
}

void CPUKernel::RenderKernel(const shared_ptr<Camera>& camera, const vector<Sphere>& spheres, const vector<Light*>& lights, RenderOption& option)
{
	KernelOption kernelOption;
	kernelOption.maxDepth = 10;
	kernelOption.frame = option.frame;
	kernelOption.enableDof = option.enableDof;
	kernelOption.width = option.width;
	kernelOption.height = option.height;
	kernelOption.xUnit = option.width / option.loopX;
	kernelOption.yUnit = option.height / option.loopY;

	kernelOption.materials.emplace_back();
	kernelOption.materials.emplace_back(DIFF, vec3(1), vec3(10.3f));
	kernelOption.materials.emplace_back(SPEC);
	kernelOption.materials.emplace_back(TRANS);
	kernelOption.materials.emplace_back(DIFF, vec3(0.75f, 0.75f, 0.75f));
	kernelOption.materials.emplace_back(DIFF, vec3(0.25f, 0.25f, 0.75f));
	kernelOption.materials.emplace_back(DIFF, vec3(0.75f, 0.25f, 0.25f));

	kernelOption.spheres = spheres;
	kernelOption.lights = lights;
	kernelOption.previousFrameData = option.previousFrameData;
	kernelOption.data = option.data;

	for (int y = 0; y < option.loopY; y++)
	{
		for (int x = 0; x < option.loopX; x++)
		{
			kernelOption.xIndex = x;
			kernelOption.yIndex = y;
			PathAccumulateKernel(camera.get(), kernelOption);
		}
	}
	
	delete option.previousFrameData;
	option.previousFrameData = kernelOption.data;
}

CPUKernel::SphereLight::SphereLight(const vec3& position, const float& radius, const float& luminance)
{
	this->position = position;
	this->radius = radius;
	this->luminance = luminance;
}

const glm::vec3& CPUKernel::SphereLight::GetPosition()
{
	return position;
}

const glm::vec3 CPUKernel::SphereLight::GetRandomPointOnSurface()
{
	float theta = two_pi<float>() * fRandom();
	float phi = acos(2 * fRandom() - 1);

	vec3 randomPoint;
	randomPoint.x = cos(theta) * sin(phi) * radius;
	randomPoint.y = sin(theta) * sin(phi) * radius;
	randomPoint.z = cos(phi) * radius;

	return randomPoint;
}

const float& CPUKernel::SphereLight::GetLuminance()
{
	return luminance;
}

ObjectIntersection CPUKernel::SphereLight::Intersect(const Ray &ray)
{
	bool hit = false;
	float distance = 0, t = 0;
	vec3 normal = vec3(0, 0, 0);
	vec3 op = position - ray.origin;
	float b = dot(op, ray.direction);
	float det = b * b - dot(op, op) + radius * radius;

	if (det < EPSILON)
		return ObjectIntersection(hit, t, normal, 1);
	else
		det = glm::sqrt(det);

	distance = (t = b - det) > EPSILON ? t : ((t = b + det) > EPSILON ? t : 0);
	if (distance > EPSILON)
	{
		hit = true;
		normal = normalize(ray.direction * distance - op);
	}
	ObjectIntersection result = ObjectIntersection(hit, distance, normal, 1, true);
	return result;
}
