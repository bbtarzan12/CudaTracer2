#include "PathKernel.cuh"

unsigned int kernelSeed;

unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ ObjectIntersection Intersect(Ray ray, KernelArray<Sphere> spheres)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp = ObjectIntersection();

	for (int i = 0; i < spheres.size; i++)
	{
		temp = spheres.array[i].Intersect(ray);

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

__device__ Ray GetReflectedRay(Ray ray, vec3 hitPoint, glm::vec3 normal, vec3 &mask, Material material, curandState* randState)
{
	switch (material.type)
	{
		case DIFF:
		{
			vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
			float r1 = two_pi<float>() * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float r2s = sqrt(r2);

			vec3 w = nl;
			vec3 u;
			if (fabs(w.x) > 0.1f)
				u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
			else
				u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
			vec3 v = cross(w, u);
			vec3 reflected = normalize((u * __cosf(r1) * r2s + v * __sinf(r1) * r2s + w * sqrt(1 - r2)));
			mask *= material.color;
			return Ray(hitPoint, reflected);
		}
		case GLOSS:
		{
			float phi = 2 * pi<float>() * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float cosTheta = __powf(1 - r2, 1.0f / (20 + 1));
			float sinTheta = __sinf(1 - cosTheta * cosTheta);

			vec3 w = normalize(ray.direction - normal * 2.0f * dot(normal, ray.direction));
			vec3 u = normalize(cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
			vec3 v = cross(w, u);

			vec3 reflected = normalize(u * __cosf(phi) * sinTheta + v * __sinf(phi) * sinTheta + w * cosTheta);
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

			if (curand_uniform(randState) < P)
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

__device__ vec3 TraceRay(Ray ray, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, curandState* randState)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < 100; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres);

		if (intersection.hit == 0)
		{
			return resultColor;
		}

		vec3 hitPoint = ray.origin + ray.direction * intersection.t;
		Material hitMaterial = materials.array[intersection.materialID];
		vec3 emission = hitMaterial.emission;

		float maxReflection = max(max(mask.r, mask.g), mask.b);
		if (curand_uniform(randState) > maxReflection)
			break;

		resultColor += mask * emission;
		ray = GetReflectedRay(ray, hitPoint, intersection.normal, mask, hitMaterial, randState);
		mask *= 1 / maxReflection;
	}
	return resultColor;
}

__device__ Ray GetRay(Camera* camera, int x, int y, bool enableDof, curandState* randState)
{
	float jitterValueX = curand_uniform(randState) - 0.5f;
	float jitterValueY = curand_uniform(randState) - 0.5f;

	vec3 wDir = glm::normalize(-camera->forward);
	vec3 uDir = glm::normalize(cross(camera->up, wDir));
	vec3 vDir = glm::cross(wDir, -uDir);

	float top = tan(camera->fov * glm::pi<float>() / 360.0f);
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
			float r1 = curand_uniform(randState);
			float r2 = curand_uniform(randState);

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

__global__ void PathAccumulateKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, cudaSurfaceObject_t surface)
{
	int width = camera->width;
	int height = camera->height;
	int x = gridDim.x * blockDim.x * option.loopX + blockIdx.x * blockDim.x + threadIdx.x;
	int y = gridDim.y * blockDim.y * option.loopY + blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int i = y * width + x;

	if (i >= width * height) return;
	curandState randState;
	float4 originColor;
	surf2Dread(&originColor, surface, x * sizeof(float4), y);

	vec3 resultColor = vec3(0, 0, 0);
	curand_init(WangHash(threadId) + WangHash(option.frame) + WangHash(option.seed), 0, 0, &randState);
	Ray ray = GetRay(camera, x, y, option.enableDof, &randState);
	vec3 color = TraceRay(ray, spheres, materials, option, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float) (option.frame - 1) + color) / (float) option.frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

__global__ void PathImageKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, cudaSurfaceObject_t surface)
{
	int width = camera->width;
	int height = camera->height;
	int x = gridDim.x * blockDim.x * option.loopX + blockIdx.x * blockDim.x + threadIdx.x;
	int y = gridDim.y * blockDim.y * option.loopY + blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int i = y * width + x;

	if (i >= width * height) return;
	curandState randState;

	vec3 resultColor = vec3(0, 0, 0);
	for (int s = 0; s < option.maxSamples; s++)
	{
		curand_init(WangHash(threadId) + WangHash(option.frame) + WangHash(option.seed) + WangHash(s), 0, 0, &randState);
		Ray ray = GetRay(camera, x, y, option.enableDof, &randState);
		resultColor += TraceRay(ray, spheres, materials, option, &randState);
	}
	resultColor /= option.maxSamples;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, const thrust::host_vector<Material>& materials, const RenderOption& option)
{
	int width = camera->width;
	int height = camera->height;
	float memoryAllocTime, renderingTime;

	cudaEvent_t start, stop;
	gpuErrorCheck(cudaEventCreate(&start));
	gpuErrorCheck(cudaEventRecord(start, 0));

	Camera* cudaCamera;
	gpuErrorCheck(cudaMalloc(&cudaCamera, sizeof(Camera)));
	gpuErrorCheck(cudaMemcpy(cudaCamera, camera.get(), sizeof(Camera), cudaMemcpyHostToDevice));

	thrust::device_vector<Sphere> cudaSpheres(spheres);
	thrust::device_vector<Material> cudaMaterials(materials);

	gpuErrorCheck(cudaEventCreate(&stop));
	gpuErrorCheck(cudaEventRecord(stop, 0));
	gpuErrorCheck(cudaEventSynchronize(stop));
	gpuErrorCheck(cudaEventElapsedTime(&memoryAllocTime, start, stop));
	gpuErrorCheck(cudaEventDestroy(start));
	gpuErrorCheck(cudaEventDestroy(stop));

	dim3 block = dim3(16, 9);
	dim3 grid = dim3(ceil(ceil(width / option.loopX) / block.x), ceil(ceil(height / option.loopY) / block.y));

	gpuErrorCheck(cudaEventCreate(&start));
	gpuErrorCheck(cudaEventRecord(start, 0));

	for (int i = 0; i < option.loopX; i++)
	{
		for (int j = 0; j < option.loopY; j++)
		{
			kernelSeed = WangHash(kernelSeed);
			KernelOption kernelOption;
			kernelOption.enableDof = option.enableDof;
			kernelOption.frame = option.frame;
			kernelOption.loopX = i;
			kernelOption.loopY = j;
			kernelOption.seed = kernelSeed;
			kernelOption.maxSamples = option.maxSamples;
			if (option.isAccumulate)
			{
				PathAccumulateKernel << <grid, block >> > (cudaCamera, ConvertToKernel(cudaSpheres), ConvertToKernel(cudaMaterials), kernelOption, option.surf);
			}
			else
			{
				PathImageKernel << <grid, block >> > (cudaCamera, ConvertToKernel(cudaSpheres), ConvertToKernel(cudaMaterials), kernelOption, option.surf);
			}
			gpuErrorCheck(cudaDeviceSynchronize());
		}
	}

	gpuErrorCheck(cudaEventCreate(&stop));
	gpuErrorCheck(cudaEventRecord(stop, 0));
	gpuErrorCheck(cudaEventSynchronize(stop));
	gpuErrorCheck(cudaEventElapsedTime(&renderingTime, start, stop));
	gpuErrorCheck(cudaEventDestroy(start));
	gpuErrorCheck(cudaEventDestroy(stop));

	gpuErrorCheck(cudaFree(cudaCamera));
}