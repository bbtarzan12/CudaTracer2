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

__device__ glm::vec3 gpuComputeTriNormal(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3)
{
	glm::vec3 u = p2 - p1;
	glm::vec3 v = p3 - p1;

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;

	return glm::normalize(glm::vec3(nx, ny, nz));
}

__device__ bool gpuTriIntersect(Ray ray, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t, glm::vec3 &normal)
{
	glm::vec3 e1, e2, h, s, q;
	float a, f, u, v;

	e1 = v1 - v0;
	e2 = v2 - v0;

	h = glm::cross(ray.direction, e2);
	a = glm::dot(e1, h);

	if (a > -0.00001f && a < 0.00001f)
	{
		return false;
	}

	f = 1.0f / a;
	s = ray.origin - v0;
	u = f * glm::dot(s, h);

	if (u < 0.0f || u > 1.0f)
	{
		return false;
	}

	q = glm::cross(s, e1);
	v = f * glm::dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f)
	{
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	t = f * glm::dot(e2, q);

	if (t > EPSILON)
	{ // ray intersection
		normal = gpuComputeTriNormal(v0, v1, v2);
		return true;
	}
	else
	{ // this means that there is a line intersection but not a ray intersection
		return false;
	}
}

__device__ bool gpuIsPointToLeftOfSplittingPlane(KDTreeNode node, const glm::vec3 &p)
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

__device__ int gpuGetNeighboringNodeIndex(KDTreeNode node, glm::vec3 p)
{

	const float fabsEpsilon = 0.000001;

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

__device__ bool gpuAABBIntersect(boundingBox bbox, Ray ray, float &t_near, float &t_far)
{
	glm::vec3 dirfrac(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

	float t1 = (bbox.min.x - ray.origin.x) * dirfrac.x;
	float t2 = (bbox.max.x - ray.origin.x) * dirfrac.x;
	float t3 = (bbox.min.y - ray.origin.y) * dirfrac.y;
	float t4 = (bbox.max.y - ray.origin.y) * dirfrac.y;
	float t5 = (bbox.min.z - ray.origin.z) * dirfrac.z;
	float t6 = (bbox.max.z - ray.origin.z) * dirfrac.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
	if (tmax < 0.0f)
	{
		return false;
	}

	// If tmin > tmax, ray does not intersect AABB.
	if (tmin > tmax)
	{
		return false;
	}

	t_near = tmin;
	t_far = tmax;
	return true;

	//vec3 minBox = bbox.min;
	//vec3 maxBox = bbox.max;

	//if (ray.direction.x < 0)
	//{
	//	ray.origin.x = minBox.x + maxBox.x - ray.origin.x;
	//	ray.direction.x = -ray.direction.x;
	//}
	//if (ray.direction.y < 0)
	//{
	//	ray.origin.y = minBox.y + maxBox.y - ray.origin.y;
	//	ray.direction.y = -ray.direction.y;
	//}
	//if (ray.direction.z < 0)
	//{
	//	ray.origin.z = minBox.z + maxBox.z - ray.origin.z;
	//	ray.direction.z = -ray.direction.z;
	//}

	//vec3 div = 1.0f / ray.direction;
	//vec3 tMin = (minBox - ray.origin) * div;
	//vec3 tMax = (maxBox - ray.origin) * div;

	//float tmin = max(max(tMin.x, tMin.y), tMin.z);
	//float tmax = min(min(tMax.x, tMax.y), tMax.z);

	//if (tmin <= tmax)
	//{
	//	t_near = tmin;
	//	t_far = tmax;
	//	return true;
	//}
	//else
	//	return false;
}

__device__ ObjectIntersection StacklessIntersect(Ray ray, int root_index, KDTreeNode *tree_nodes, int *kd_tri_index_list, glm::ivec3 *tris, glm::vec3 *verts)
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
		glm::vec3 p_entry = ray.origin + (t_entry * ray.direction);
		while (!curr_node.is_leaf_node)
		{
			curr_node = gpuIsPointToLeftOfSplittingPlane(curr_node, p_entry) ? tree_nodes[curr_node.left_child_index] : tree_nodes[curr_node.right_child_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
		for (int i = curr_node.first_tri_index; i < (curr_node.first_tri_index + curr_node.num_tris); ++i)
		{
			glm::vec3 tri = tris[kd_tri_index_list[i]];
			glm::vec3 v0 = verts[(int) tri[0]];
			glm::vec3 v1 = verts[(int) tri[1]];
			glm::vec3 v2 = verts[(int) tri[2]];

			// Perform ray/triangle intersection test.
			float tmp_t = INF;
			glm::vec3 tmp_normal(0.0f, 0.0f, 0.0f);
			bool intersects_tri = gpuTriIntersect(ray, v0, v1, v2, tmp_t, tmp_normal);

			if (intersects_tri)
			{
				if (tmp_t < t_exit)
				{
					intersection.hit = true;
					t_exit = tmp_t;
					intersection.normal = tmp_normal;
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
		glm::vec3 p_exit = ray.origin + (t_entry * ray.direction);
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

__device__ ObjectIntersection Intersect(Ray ray, KernelArray<Sphere> spheres, glm::ivec3 *mesh_tris, glm::vec3 *mesh_verts, int kd_tree_root_index, KDTreeNode *kd_tree_nodes, int *kd_tree_tri_indices)
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

	ObjectIntersection meshIntersect = StacklessIntersect(ray, kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices, mesh_tris, mesh_verts);

	if (meshIntersect.hit && meshIntersect.t < intersection.t)
	{
		intersection = meshIntersect;
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

__device__ vec3 TraceRay(Ray ray, KernelArray<Sphere> spheres, KernelArray<Material> materials, glm::ivec3 *mesh_tris, glm::vec3 *mesh_verts, int kd_tree_root_index, KDTreeNode *kd_tree_nodes, int *kd_tree_tri_indices, KernelOption option, curandState* randState)
{
	vec3 resultColor = vec3(0);
	vec3 mask = vec3(1);

	for (int depth = 0; depth < 100; depth++)
	{
		ObjectIntersection intersection = Intersect(ray, spheres, mesh_tris, mesh_verts, kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices);

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

__global__ void PathAccumulateKernel(Camera* camera, KernelArray<Sphere> spheres, KernelArray<Material> materials, glm::ivec3 *mesh_tris, glm::vec3 *mesh_verts, int kd_tree_root_index, KDTreeNode *kd_tree_nodes, int *kd_tree_tri_indices, KernelOption option, cudaSurfaceObject_t surface)
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
	vec3 color = TraceRay(ray, spheres, materials, mesh_tris, mesh_verts, kd_tree_root_index, kd_tree_nodes, kd_tree_tri_indices, option, &randState);
	resultColor = (vec3(originColor.x, originColor.y, originColor.z) * (float) (option.frame - 1) + color) / (float) option.frame;
	surf2Dwrite(make_float4(resultColor.r, resultColor.g, resultColor.b, 1.0f), surface, x * sizeof(float4), y);
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

void RenderKernel(const shared_ptr<Camera>& camera, const thrust::host_vector<Sphere>& spheres, const std::vector<Mesh*> meshes, const KDTree* tree, const thrust::host_vector<Material>& materials, const RenderOption& option)
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
	glm::ivec3 *cuda_mesh_tris;
	gpuErrorCheck(cudaMalloc((void**) &cuda_mesh_tris, tree->getMeshTris().size() * sizeof(glm::ivec3)));
	gpuErrorCheck(cudaMemcpy(cuda_mesh_tris, tree->getMeshTris().data(), tree->getMeshTris().size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice));

	// Send mesh vertices to GPU.
	glm::vec3 *cuda_mesh_verts;
	gpuErrorCheck(cudaMalloc((void**) &cuda_mesh_verts, tree->getMeshVerts().size() * sizeof(glm::vec3)));
	gpuErrorCheck(cudaMemcpy(cuda_mesh_verts, tree->getMeshVerts().data(), tree->getMeshVerts().size() * sizeof(glm::vec3), cudaMemcpyHostToDevice));

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
			if (option.isAccumulate)
			{
				PathAccumulateKernel << <grid, block >> > (cudaCamera, ConvertToKernel(cudaSpheres), ConvertToKernel(cudaMaterials), cuda_mesh_tris, cuda_mesh_verts, tree->getRootIndex(), cuda_kd_tree_nodes, cuda_kd_tree_tri_indices, kernelOption, option.surf);
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

	gpuErrorCheck(cudaFree(cuda_mesh_tris));
	gpuErrorCheck(cudaFree(cuda_mesh_verts));
	gpuErrorCheck(cudaFree(cuda_kd_tree_nodes));
	gpuErrorCheck(cudaFree(cuda_kd_tree_tri_indices));
	delete[] tri_index_array;
}