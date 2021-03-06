#ifndef H_MESH
#define H_MESH

#include <glm.hpp>
#include <vector>
#include "cuda_runtime.h"

using namespace std;
using namespace glm;

enum MaterialType { NONE, DIFF, GLOSS, TRANS, SPEC, MERGE };

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
	float specular;
	float metalic;
	bool isTransparent = false;
	float nc = 1.0f;
	float nt = 1.5f;
};

class Mesh
{
public:
	Mesh(string filePath);
	~Mesh();

	string name;
	vector<vec3> verts;
	vector<vec3> norms;
	vector<vec2> uvs;
	vector<Material> materials;
	vector<ivec3> vertexIndices;
	vector<ivec3> normalIndices;
	vector<int> materialIndices;

	unsigned int vbo, vao;
	int bufferSize;
	vector<float> buffer;
};

#endif