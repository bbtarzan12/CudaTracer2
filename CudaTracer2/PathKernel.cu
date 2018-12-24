#include "PathKernel.cuh"

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

	const float fabsEpsilon = 0.000005;

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
	tmin = (bbox.min.x - ray.origin.x) / ray.direction.x;
	tmax = (bbox.max.x - ray.origin.x) / ray.direction.x;

	if (tmin > tmax) thrust::swap(tmin, tmax);

	float tymin = (bbox.min.y - ray.origin.y) / ray.direction.y;
	float tymax = (bbox.max.y - ray.origin.y) / ray.direction.y;

	if (tymin > tymax) thrust::swap(tymin, tymax);

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (bbox.min.z - ray.origin.z) / ray.direction.z;
	float tzmax = (bbox.max.z - ray.origin.z) / ray.direction.z;

	if (tzmin > tzmax) thrust::swap(tzmin, tzmax);

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return true;
}

__device__ ObjectIntersection StacklessIntersect(Ray ray, int root_index, KDTreeNode *tree_nodes, int *kd_tri_index_list, ivec3 *vertexIndices, ivec3* normalIndices, vec3 *verts, vec3* norms)
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
			ivec3 tri = vertexIndices[kd_tri_index_list[i]];
			ivec3 norm = normalIndices[kd_tri_index_list[i]];
			vec3 v0 = verts[tri.x];
			vec3 v1 = verts[tri.y];
			vec3 v2 = verts[tri.z];
			vec3 n0 = norms[norm.x];
			vec3 n1 = norms[norm.y];
			vec3 n2 = norms[norm.z];

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
					intersection.normal = tmp_normal;
					intersection.materialID = 2;
				}
			}
		}

		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = gpuAABBIntersect(curr_node.bbox, ray, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box)
		{
			// Set t_entry to be the entrance point of the next (neighboring) node.
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
	intersection.t = t_exit;
	return intersection;
}

__device__ ObjectIntersection Intersect(Ray ray, KernelArray<Sphere> spheres, ivec3 *vertexIndices, ivec3* normalIndices, vec3 *verts, vec3* norms, int kd_tree_root_index, KDTreeNode *kd_tree_nodes, int *kd_tree_tri_indices)
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

	ObjectIntersection meshIntersect = StacklessIntersect(ray, kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices, vertexIndices, normalIndices, verts, norms);

	if (meshIntersect.hit && meshIntersect.t < intersection.t)
	{
		intersection = meshIntersect;
	}

	return intersection;
}

__device__ Ray GetReflectedRay(Ray ray, vec3 hitPoint, vec3 normal, vec3 &mask, Material material, curandState* randState)
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

__device__ vec3 TraceRay(Ray ray, KernelOption option, curandState* randState)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < 5; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, option.spheres, option.vertexIndices, option.normalIndices, option.verts, option.norms, option.kdTreeRootIndex, option.kdTreeNodes, option.kdTreeTriIndices);

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
		Material hitMaterial = option.materials.array[intersection.materialID];
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

__global__ void PathImageKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, KernelOption option, cudaSurfaceObject_t surface)
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
	dst = FreeImage_ToneMapping(src, FITMO_REINHARD05);
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

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, const KDTree* tree, const thrust::host_vector<Material>& materials, const RenderOption& option)
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

	// Send mesh triangles to GPU.
	ivec3 *cuda_vert_indices;
	gpuErrorCheck(cudaMalloc((void**) &cuda_vert_indices, tree->getVertexIndices().size() * sizeof(ivec3)));
	gpuErrorCheck(cudaMemcpy(cuda_vert_indices, tree->getVertexIndices().data(), tree->getVertexIndices().size() * sizeof(ivec3), cudaMemcpyHostToDevice));

	ivec3 *cuda_normal_indices;
	gpuErrorCheck(cudaMalloc((void**)&cuda_normal_indices, tree->getNormalIndices().size() * sizeof(ivec3)));
	gpuErrorCheck(cudaMemcpy(cuda_normal_indices, tree->getNormalIndices().data(), tree->getNormalIndices().size() * sizeof(ivec3), cudaMemcpyHostToDevice));


	// Send mesh vertices to GPU.
	vec3 *cuda_mesh_verts;
	gpuErrorCheck(cudaMalloc((void**) &cuda_mesh_verts, tree->getMeshVerts().size() * sizeof(vec3)));
	gpuErrorCheck(cudaMemcpy(cuda_mesh_verts, tree->getMeshVerts().data(), tree->getMeshVerts().size() * sizeof(vec3), cudaMemcpyHostToDevice));

	// Send mesh normals to GPU.
	vec3 *cuda_mesh_norms;
	gpuErrorCheck(cudaMalloc((void**)&cuda_mesh_norms, tree->getMeshNorms().size() * sizeof(vec3)));
	gpuErrorCheck(cudaMemcpy(cuda_mesh_norms, tree->getMeshNorms().data(), tree->getMeshNorms().size() * sizeof(vec3), cudaMemcpyHostToDevice));

	// Send kd-tree nodes to GPU.
	KDTreeNode *cuda_kd_tree_nodes;
	gpuErrorCheck(cudaMalloc((void**) &cuda_kd_tree_nodes, tree->getNumNodes() * sizeof(KDTreeNode)));
	gpuErrorCheck(cudaMemcpy(cuda_kd_tree_nodes, tree->getTreeNodes().data(), tree->getNumNodes() * sizeof(KDTreeNode), cudaMemcpyHostToDevice));

	std::vector<int> kd_tree_tri_indics = tree->getTriIndexList();
	int *tri_index_array = new int[kd_tree_tri_indics.size()];
	for (int i = 0; i < kd_tree_tri_indics.size(); ++i)
	{
		tri_index_array[i] = kd_tree_tri_indics[i];
	}

	// Send kd-tree triangle indices to GPU.
	int *cuda_kd_tree_tri_indices;
	gpuErrorCheck(cudaMalloc((void**) &cuda_kd_tree_tri_indices, kd_tree_tri_indics.size() * sizeof(int)));
	gpuErrorCheck(cudaMemcpy(cuda_kd_tree_tri_indices, tri_index_array, kd_tree_tri_indics.size() * sizeof(int), cudaMemcpyHostToDevice));

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
			kernelOption.spheres = ConvertToKernel(cudaSpheres);
			kernelOption.materials = ConvertToKernel(cudaMaterials);
			kernelOption.vertexIndices = cuda_vert_indices;
			kernelOption.normalIndices = cuda_normal_indices;
			kernelOption.verts = cuda_mesh_verts;
			kernelOption.norms = cuda_mesh_norms;
			kernelOption.kdTreeRootIndex = tree->getRootIndex();
			kernelOption.kdTreeNodes = cuda_kd_tree_nodes;
			kernelOption.kdTreeTriIndices = cuda_kd_tree_tri_indices;
			kernelOption.surface = option.surf;
			if (option.isAccumulate)
			{
				PathAccumulateKernel << <grid, block >> > (cudaCamera, kernelOption);
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

	gpuErrorCheck(cudaFree(cuda_vert_indices));
	gpuErrorCheck(cudaFree(cuda_normal_indices));

	gpuErrorCheck(cudaFree(cuda_mesh_verts));
	gpuErrorCheck(cudaFree(cuda_mesh_norms));
	gpuErrorCheck(cudaFree(cuda_kd_tree_nodes));
	gpuErrorCheck(cudaFree(cuda_kd_tree_tri_indices));
	delete[] tri_index_array;
}