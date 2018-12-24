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

	vec3 position;
	vector<vec3> verts;
	vector<vec3> norms;
	vector<vec2> uvs;
	vector<ivec3> tris;
};

#endif