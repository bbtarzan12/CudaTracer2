#include "PathKernel.cuh"
#include <thrust/device_ptr.h>
#include <FreeImage.h>

unsigned int kernelSeed;

// HDR Texture
texture<float4, 1, cudaReadModeElementType> HDRtexture;
float4* cudaHDRmap;
unsigned int hdrHeight, hdrWidth;

__host__ __device__ unsigned int WangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ vec3 GetConeSample(vec3 dir, float extent, curandState* randState)
{
	dir = normalize(dir);
	vec3 o1 = normalize(abs(dir.x) > abs(dir.z) ? vec3(-dir.y, dir.x, 0.0) : vec3(0.0, -dir.z, dir.y));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = vec2(curand_uniform(randState), curand_uniform(randState));
	r.x = r.x * two_pi<float>();
	r.y = 1.0 - r.y * extent;
	float oneminus = sqrt(1.0 - r.y*r.y);
	return normalize(cos(r.x)*oneminus*o1 + sin(r.x)*oneminus*o2 + r.y*dir);
}

__device__ bool gpuTriIntersect(Ray ray, vec3 v0, vec3 v1, vec3 v2, vec3 n0, vec3 n1, vec3 n2, float &t, vec3 &normal)
{
	vec3 e1, e2, h, s, q;
	float a, f, u, v;

	e1 = v1 - v0;
	e2 = v2 - v0;

	h = cross(ray.direction, e2);
	a = dot(e1, h);

	if (a > -0.00001f && a < 0.00001f)
	{
		return false;
	}

	f = 1.0f / a;
	s = ray.origin - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f)
	{
		return false;
	}

	q = cross(s, e1);
	v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f)
	{
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	t = f * dot(e2, q);

	if (t > EPSILON)
	{ // ray intersection
		normal = normalize((1 - u - v) * n0 + u * n1 + v * n2);
		return true;
	}
	else
	{ // this means that there is a line intersection but not a ray intersection
		return false;
	}
}

__device__ bool gpuIsPointToLeftOfSplittingPlane(KDTreeNode node, const vec3 &p)
{
	if (node.split_plane_axis == X_AXIS)
	{
		return (p.x <= node.split_plane_value);
	}
	else if (node.split_plane_axis == Y_AXIS)
	{
		return (p.y <= node.split_plane_value);
	}
	else if (node.split_plane_axis == Z_AXIS)
	{
		return (p.z <= node.split_plane_value);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else
	{
		return false;
	}
}

__device__ int gpuGetNeighboringNodeIndex(KDTreeNode node, vec3 p)
{

	const float fabsEpsilon = 0.0001f;

	// Check left face.
	if (fabs(p.x - node.bbox.min.x) < fabsEpsilon)
	{
		return node.neighbor_node_indices[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - node.bbox.max.z) < fabsEpsilon)
	{
		return node.neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - node.bbox.max.x) < fabsEpsilon)
	{
		return node.neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - node.bbox.min.z) < fabsEpsilon)
	{
		return node.neighbor_node_indices[BACK];
	}
	// Check top face.
	else if (fabs(p.y - node.bbox.max.y) < fabsEpsilon)
	{
		return node.neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - node.bbox.min.y) < fabsEpsilon)
	{
		return node.neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else
	{
		return -1;
	}
}

__device__ bool gpuAABBIntersect(boundingBox bbox, Ray ray, float &tmin, float &tmax)
{
	float tymin, tymax, tzmin, tzmax;

	tmin = (bbox[ray.sign[0]].x - ray.origin.x) * ray.invdir.x;
	tmax = (bbox[1 - ray.sign[0]].x - ray.origin.x) * ray.invdir.x;
	tymin = (bbox[ray.sign[1]].y - ray.origin.y) * ray.invdir.y;
	tymax = (bbox[1 - ray.sign[1]].y - ray.origin.y) * ray.invdir.y;

	if ((tmin > tymax) || (tymin > tmax))
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;

	tzmin = (bbox[ray.sign[2]].z - ray.origin.z) * ray.invdir.z;
	tzmax = (bbox[1 - ray.sign[2]].z - ray.origin.z) * ray.invdir.z;

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;

	return true;

	//for (int a = 0; a < 3; a++)
	//{
	//	float invD = 1.0f / ray.direction[a];
	//	float t0 = (bbox.min[a] - ray.origin[a]) * invD;
	//	float t1 = (bbox.max[a] - ray.origin[a]) * invD;
	//	if (invD < 0.0f)
	//	{
	//		float tmp = t0;
	//		t0 = t1;
	//		t1 = tmp;
	//	}
	//	tmin = t0 > tmin ? t0 : tmin;
	//	tmax = t1 < tmax ? t1 : tmax;
	//	if (tmax <= tmin)
	//		return false;
	//}
	//return true;
}

__device__ ObjectIntersection StacklessIntersect(Ray ray, int root_index, KDTreeNode *tree_nodes, int *kd_tri_index_list, ivec3 *vertexIndices, ivec3* normalIndices, int* materialIndices, vec3 *verts, vec3* norms, Material* materials)
{
	KDTreeNode curr_node = tree_nodes[root_index];
	ObjectIntersection intersection = ObjectIntersection();

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;
	bool intersects_root_node_bounding_box = gpuAABBIntersect(curr_node.bbox, ray, t_entry, t_exit);

	if (!intersects_root_node_bounding_box)
	{
		return false;
	}


	float t_entry_prev = -INF;
	while (t_entry < t_exit && t_entry > t_entry_prev)
	{
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		vec3 p_entry = ray.origin + (t_entry * ray.direction);
		while (!curr_node.is_leaf_node)
		{
			curr_node = gpuIsPointToLeftOfSplittingPlane(curr_node, p_entry) ? tree_nodes[curr_node.left_child_index] : tree_nodes[curr_node.right_child_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
		for (int i = curr_node.first_tri_index; i < (curr_node.first_tri_index + curr_node.num_tris); ++i)
		{
			ivec3& tri = vertexIndices[kd_tri_index_list[i]];
			ivec3& norm = normalIndices[kd_tri_index_list[i]];
			int& material = materialIndices[kd_tri_index_list[i]];
			vec3& v0 = verts[tri.x];
			vec3& v1 = verts[tri.y];
			vec3& v2 = verts[tri.z];
			vec3& n0 = norms[norm.x];
			vec3& n1 = norms[norm.y];
			vec3& n2 = norms[norm.z];

			// Perform ray/triangle intersection test.
			float tmp_t = INF;
			vec3 tmp_normal(0.0f, 0.0f, 0.0f);
			bool intersects_tri = gpuTriIntersect(ray, v0, v1, v2, n0, n1, n2, tmp_t, tmp_normal);

			if (intersects_tri)
			{
				if (tmp_t < t_exit)
				{
					intersection.hit = true;
					t_exit = tmp_t;
					intersection.t = t_exit;
					intersection.normal = tmp_normal;
					intersection.materialID = material;
				}
			}
		}

		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = gpuAABBIntersect(curr_node.bbox, ray, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box)
		{
			// Set t_entry to be the entrance point of the next (neighboring) node.d
			t_entry = tmp_t_far;
		}
		else
		{
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}

		// Get neighboring node using ropes attached to current node.
		vec3 p_exit = ray.origin + (t_entry * ray.direction);
		int new_node_index = gpuGetNeighboringNodeIndex(curr_node, p_exit);

		// Break if neighboring node not found, meaning we've exited the kd-tree.
		if (new_node_index == -1)
		{
			break;
		}

		curr_node = tree_nodes[new_node_index];
	}
	return intersection;
}

__device__ ObjectIntersection Intersect(Ray ray, KernelArray<Sphere> spheres, ivec3 *vertexIndices, ivec3* normalIndices, int* materialIndices, vec3 *verts, vec3* norms, Material* materials, int kd_tree_root_index, KDTreeNode *kd_tree_nodes, int *kd_tree_tri_indices)
{
	ObjectIntersection intersection = ObjectIntersection();
	ObjectIntersection temp = ObjectIntersection();

	intersection.t = INF;

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

	ObjectIntersection meshIntersect = StacklessIntersect(ray, kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices, vertexIndices, normalIndices, materialIndices,  verts, norms, materials);

	if (meshIntersect.hit && meshIntersect.t < intersection.t)
	{
		intersection = meshIntersect;
	}

	return intersection;
}

__device__ Ray GetReflectedRay(Ray ray, vec3 hitPoint, vec3 normal, vec3 &mask, Material material, curandState* randState, float specular, float metalic, bool isTransparent, float nc = 1.0f, float nt = 1.5f)
{
	switch (material.type)
	{
		case MERGE:
		{
			if (isTransparent)
			{
				vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
				vec3 reflection = ray.direction - normal * 2.0f * dot(normal, ray.direction);
				bool into = dot(normal, nl) > EPSILON;
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
			else
			{
				float phi = 2 * pi<float>() * curand_uniform(randState);
				float r2 = curand_uniform(randState);
				float cosTheta = powf(1 - r2, 1.0f / (metalic + 1));
				float sinTheta = sqrt(1 - cosTheta * cosTheta);

				vec3 w = normalize(ray.direction - normal * 2.0f * dot(normal, ray.direction));
				vec3 u = normalize(cross((fabs(w.x) > .1 ? vec3(0, 1, 0) : vec3(1, 0, 0)), w));
				vec3 v = cross(w, u);

				vec3 reflected = normalize((u * __cosf(phi) * sinTheta + v * __sinf(phi) * sinTheta) * (1 - specular) + w * cosTheta);
				mask *= material.color;
				return Ray(hitPoint, reflected);
			}
		}
		case DIFF:
		{
			vec3 nl = dot(normal, ray.direction) < EPSILON ? normal : normal * -1.0f;
			float phi = two_pi<float>() * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float r2s = sqrt(r2);

			vec3 w = nl;
			vec3 u;
			if (fabs(w.x) > 0.1f)
				u = normalize(cross(vec3(0.0f, 1.0f, 0.0f), w));
			else
				u = normalize(cross(vec3(1.0f, 0.0f, 0.0f), w));
			vec3 v = cross(w, u);
			vec3 reflected = normalize((u * __cosf(phi) * r2s + v * __sinf(phi) * r2s + w * sqrt(1 - r2)));

			mask *= material.color;
			return Ray(hitPoint, reflected);
		}
		case GLOSS:
		{
			float phi = 2 * pi<float>() * curand_uniform(randState);
			float r2 = curand_uniform(randState);
			float cosTheta = powf(1 - r2, 1.0f / (20 + 1));
			float sinTheta = sqrt(1 - cosTheta * cosTheta);

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

__device__ vec3 TraceRay(Ray ray, KernelOption option, curandState* randState)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < option.maxDepth; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, option.spheres, option.vertexIndices, option.normalIndices, option.materialIndices, option.verts, option.norms, option.materials, option.kdTreeRootIndex, option.kdTreeNodes, option.kdTreeTriIndices);

		if (!intersection.hit)
		{
			float longlatX = atan2(ray.direction.x, ray.direction.z);
			longlatX = longlatX < EPSILON ? longlatX + two_pi<float>() : longlatX;
			float longlatY = acos(-ray.direction.y);

			float u = longlatX / two_pi<float>();
			float v = longlatY / pi<float>();

			int u2 = (int)(u * option.hdrWidth);
			int tvec = (int)(v * option.hdrHeight);

			int HDRtexelidx = u2 + tvec * option.hdrWidth;

			float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);
			vec3 HDRcol2 = vec3(HDRcol.x, HDRcol.y, HDRcol.z);

			return resultColor + (mask * HDRcol2);
		}

		vec3 hitPoint = ray.origin + ray.direction * intersection.t;
		Material hitMaterial = option.materials[intersection.materialID];
		vec3 emission = hitMaterial.emission;

		float maxReflection = max(max(mask.r, mask.g), mask.b);
		if (curand_uniform(randState) > maxReflection)
			break;

		vec3 sunSampleDir = GetConeSample(option.sunDirection, option.sunExtent, randState);
		float sunLight = dot(intersection.normal, sunSampleDir);
		Ray lightRay = Ray(hitPoint + intersection.normal * EPSILON, sunSampleDir);

		ObjectIntersection lightIntersection = Intersect(lightRay, option.spheres, option.vertexIndices, option.normalIndices, option.materialIndices, option.verts, option.norms, option.materials, option.kdTreeRootIndex, option.kdTreeNodes, option.kdTreeTriIndices);
		if (sunLight > 0.0f && !lightIntersection.hit)
		{
			resultColor += sunLight * option.sunLuminance * mask;
		}

		resultColor += mask * emission;
		ray = GetReflectedRay(ray, hitPoint, intersection.normal, mask, hitMaterial, randState, hitMaterial.specular, hitMaterial.metalic, hitMaterial.isTransparent, option.nc, option.nt);
		mask *= 1 / maxReflection;
	}
	return resultColor;
}

__device__ Ray GetRay(Camera* camera, int x, int y, bool enableDof, curandState* randState)
{
	float jitterValueX = 2 * curand_uniform(randState) - 1.0f;
	float jitterValueY = 2 * curand_uniform(randState) - 1.0f;

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

__global__ void PathAccumulateKernel(Camera* camera, KernelOption option)
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
	surf2Dread(&originColor, option.surface, x * sizeof(float4), y);

	vec3 resultColor = vec3(0, 0, 0);
	curand_init(WangHash(threadId) + WangHash(option.frame) + WangHash(option.seed), 0, 0, &randState);
	Ray ray = GetRay(camera, x, y, option.enableDof, &randState);
	vec3 color = TraceRay(ray, option, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float) (option.frame - 1) + color) / (float) option.frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), option.surface, x * sizeof(float4), y);
}

__global__ void PathImageKernel(Camera* camera, KernelOption option)
{
	//int width = camera->width;
	//int height = camera->height;
	//int x = gridDim.x * blockDim.x * option.loopX + blockIdx.x * blockDim.x + threadIdx.x;
	//int y = gridDim.y * blockDim.y * option.loopY + blockIdx.y * blockDim.y + threadIdx.y;
	//int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	//int i = y * width + x;

	//if (i >= width * height) return;
	//curandState randState;

	//vec3 resultColor = vec3(0, 0, 0);
	//for (int s = 0; s < option.maxSamples; s++)
	//{
	//	curand_init(WangHash(threadId) + WangHash(option.frame) + WangHash(option.seed) + WangHash(s), 0, 0, &randState);
	//	Ray ray = GetRay(camera, x, y, option.enableDof, &randState);
	//	resultColor += TraceRay(ray, spheres, materials, option, &randState);
	//}
	//resultColor /= option.maxSamples;
	//surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
}

void InitHDRTexture(const char* hdrFileName)
{
	FREE_IMAGE_FORMAT fif = FIF_HDR;
	FIBITMAP *src(nullptr);
	FIBITMAP *dst(nullptr);
	BYTE* bits(nullptr);
	float4* cpuHDRmap;

	src = FreeImage_Load(fif, hdrFileName);
	dst = FreeImage_ToneMapping(src, FITMO_DRAGO03);
	bits = FreeImage_GetBits(dst);
	if (bits == nullptr)
		return;

	hdrHeight = FreeImage_GetHeight(src);
	hdrWidth = FreeImage_GetWidth(src);

	cpuHDRmap = new float4[ hdrHeight * hdrWidth ];

	for (int x = 0; x < hdrWidth; x++)
	{
		for (int y = 0; y < hdrHeight; y++)
		{
			RGBQUAD rgbQuad;
			FreeImage_GetPixelColor(dst, x, y, &rgbQuad);
			cpuHDRmap[y*hdrWidth + x].x = rgbQuad.rgbRed / 256.0f;
			cpuHDRmap[y*hdrWidth + x].y = rgbQuad.rgbGreen / 256.0f;
			cpuHDRmap[y*hdrWidth + x].z = rgbQuad.rgbBlue / 256.0f;
			cpuHDRmap[y*hdrWidth + x].w = 1.0f;
		}
	}

	gpuErrorCheck(cudaMalloc(&cudaHDRmap, hdrWidth * hdrHeight * sizeof(float4)));
	gpuErrorCheck(cudaMemcpy(cudaHDRmap, cpuHDRmap, hdrWidth * hdrHeight * sizeof(float4), cudaMemcpyHostToDevice));

	HDRtexture.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &HDRtexture, cudaHDRmap, &channel4desc, hdrWidth * hdrHeight * sizeof(float4));

	printf("[CUDA] Load HDR Map Success\n");
	printf("[HDR] Width : %d Height : %d\n", hdrWidth, hdrHeight);

	FreeImage_Unload(src);
	FreeImage_Unload(dst);
	delete cpuHDRmap;
}

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, KDTree* tree, const RenderOption& option)
{
	if (!tree)
		return;
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
	thrust::device_vector<ivec3> cuda_vert_indices(tree->getVertexIndices());
	thrust::device_vector<ivec3> cuda_normal_indices(tree->getNormalIndices());
	thrust::device_vector<int> cuda_material_indices(tree->getMaterialIndices());
	thrust::device_vector<vec3> cuda_mesh_verts(tree->getMeshVerts());
	thrust::device_vector<vec3> cuda_mesh_norms(tree->getMeshNorms());
	thrust::device_vector<Material> cuda_mesh_materials(tree->getMeshMaterials());
	thrust::device_vector<KDTreeNode> cuda_kd_tree_nodes(tree->getTreeNodes());
	thrust::device_vector<int> cuda_kd_tree_tri_indices = tree->getTriIndexList();

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
			kernelOption.hdrHeight = hdrHeight;
			kernelOption.hdrWidth = hdrWidth;
			kernelOption.maxDepth = 50;
			kernelOption.sunDirection = option.sunDirection;
			kernelOption.sunLuminance = option.sunLuminance;
			kernelOption.sunExtent = option.sunExtent;
			kernelOption.spheres = ConvertToKernel(cudaSpheres);
			kernelOption.vertexIndices = thrust::raw_pointer_cast(cuda_vert_indices.data());
			kernelOption.normalIndices = thrust::raw_pointer_cast(cuda_normal_indices.data());
			kernelOption.materialIndices = thrust::raw_pointer_cast(cuda_material_indices.data());
			kernelOption.verts = thrust::raw_pointer_cast(cuda_mesh_verts.data());
			kernelOption.norms = thrust::raw_pointer_cast(cuda_mesh_norms.data());
			kernelOption.materials = thrust::raw_pointer_cast(cuda_mesh_materials.data());
			kernelOption.kdTreeRootIndex = tree->getRootIndex();
			kernelOption.kdTreeNodes = thrust::raw_pointer_cast(cuda_kd_tree_nodes.data());
			kernelOption.kdTreeTriIndices = thrust::raw_pointer_cast(cuda_kd_tree_tri_indices.data());
			kernelOption.surface = option.surf;
			kernelOption.metalic = option.metalic;
			kernelOption.specular = option.specular;
			kernelOption.isTransparent = option.isTransparent;
			kernelOption.nc = option.nc;
			kernelOption.nt = option.nt;
			if (option.isAccumulate)
			{
				PathAccumulateKernel << <grid, block >> > (cudaCamera, kernelOption);
			}
			else
			{
				PathImageKernel << <grid, block >> > (cudaCamera, kernelOption);
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