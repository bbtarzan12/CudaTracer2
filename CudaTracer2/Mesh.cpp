#include "Mesh.h"

#include "MeasureTime.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#pragma region Mesh

Mesh::Mesh(vec3 position /*= vec3(0)*/, string fileName /*= ""*/)
{
	auto timer = MeasureTime::Timer();
	timer.Start("[Mesh] Load Start");
	
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;

	string warn;
	string err;
	string basePath = "./";
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fileName.c_str(), basePath.c_str(), true);

	if (!warn.empty())
	{
		cout << "WARN: " << warn << endl;
	}

	if (!err.empty())
	{
		cerr << "ERR: " << err << endl;
	}

	if (!ret)
	{
		cout << "Failed to load/parse" << fileName << endl;
	}

	for (size_t v = 0; v < attrib.vertices.size() / 3; v++)
	{
		verts.emplace_back(attrib.vertices[3 * v + 0] + position.x, attrib.vertices[3 * v + 1] + position.y, attrib.vertices[3 * v + 2] + position.z);
	}

	for (size_t v = 0; v < attrib.normals.size() / 3; v++)
	{
		norms.emplace_back(attrib.normals[3 * v + 0], attrib.normals[3 * v + 1], attrib.normals[3 * v + 2]);
	}

	for (size_t v = 0; v < attrib.texcoords.size() / 2; v++)
	{
		uvs.emplace_back(attrib.texcoords[2 * v], attrib.texcoords[2 * v + 1]);
	}

	for (auto & shape : shapes)
	{
		size_t index_offset = 0;

		materialIndices = move(shape.mesh.material_ids);
		for (auto & triangle : shape.mesh.num_face_vertices)
		{
			tinyobj::index_t idx = shape.mesh.indices[index_offset];
			tinyobj::index_t idy = shape.mesh.indices[index_offset + 1];
			tinyobj::index_t idz = shape.mesh.indices[index_offset + 2];

			vertexIndices.emplace_back(idx.vertex_index, idy.vertex_index, idz.vertex_index);
			normalIndices.emplace_back(idx.normal_index, idy.normal_index, idz.normal_index);
			index_offset += 3;
		}
	}

	for (auto& objMaterial : materials)
	{
		Material material;
		material.color = vec3(objMaterial.diffuse[0], objMaterial.diffuse[1], objMaterial.diffuse[2]);
		material.emission = vec3(objMaterial.emission[0], objMaterial.emission[1], objMaterial.emission[2]);
		this->materials.push_back(material);
	}

	timer.End("[Mesh] Load Success");
}

#pragma endregion Mesh