#ifndef H_MESH
#define H_MESH

#include <glm.hpp>
#include <vector>

using namespace std;
using namespace glm;

class Mesh
{
public:
	Mesh(vec3 position = vec3(0), std::string fileName = "", int materialID = 0);

	vector<vec3> verts;
	vector<vec3> norms;
	vector<vec2> uvs;
	vector<ivec3> vertexIndices;
	vector<ivec3> normalIndices;
};

#endif