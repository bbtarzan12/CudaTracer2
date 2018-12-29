#include "Mesh.h"

#include "MeasureTime.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <GL/glew.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>

#pragma region Mesh

Mesh::Mesh(string fileName /*= ""*/)
{
	auto timer = MeasureTime::Timer();
	timer.Start("[Mesh] Load Start");

	vector<float> buffer;
	
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;

	string warn;
	string err;

	size_t found = fileName.find_last_of("/\\");
	string base = fileName.substr(0, found);
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fileName.c_str(), base.c_str(), true);

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
		verts.emplace_back(attrib.vertices[3 * v + 0], attrib.vertices[3 * v + 1], attrib.vertices[3 * v + 2] );
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

		materialIndices.reserve(materialIndices.size() + shape.mesh.material_ids.size());
		materialIndices.insert(materialIndices.end(), shape.mesh.material_ids.begin(), shape.mesh.material_ids.end());
		for (auto & triangle : shape.mesh.num_face_vertices)
		{
			tinyobj::index_t idx = shape.mesh.indices[index_offset];
			tinyobj::index_t idy = shape.mesh.indices[index_offset + 1];
			tinyobj::index_t idz = shape.mesh.indices[index_offset + 2];

			vertexIndices.emplace_back(idx.vertex_index, idy.vertex_index, idz.vertex_index);
			normalIndices.emplace_back(idx.normal_index, idy.normal_index, idz.normal_index);
			index_offset += 3;

			buffer.emplace_back(verts[idx.vertex_index].x);
			buffer.emplace_back(verts[idx.vertex_index].y);
			buffer.emplace_back(verts[idx.vertex_index].z);

			buffer.emplace_back(norms[idx.normal_index].x);
			buffer.emplace_back(norms[idx.normal_index].y);
			buffer.emplace_back(norms[idx.normal_index].z);

			buffer.emplace_back(verts[idy.vertex_index].x);
			buffer.emplace_back(verts[idy.vertex_index].y);
			buffer.emplace_back(verts[idy.vertex_index].z);

			buffer.emplace_back(norms[idy.normal_index].x);
			buffer.emplace_back(norms[idy.normal_index].y);
			buffer.emplace_back(norms[idy.normal_index].z);

			buffer.emplace_back(verts[idz.vertex_index].x);
			buffer.emplace_back(verts[idz.vertex_index].y);
			buffer.emplace_back(verts[idz.vertex_index].z);

			buffer.emplace_back(norms[idz.normal_index].x);
			buffer.emplace_back(norms[idz.normal_index].y);
			buffer.emplace_back(norms[idz.normal_index].z);
		}
	}

	for (auto& objMaterial : materials)
	{
		Material material;
		material.type = MERGE;
		material.color = vec3(objMaterial.diffuse[0], objMaterial.diffuse[1], objMaterial.diffuse[2]);
		material.emission = vec3(objMaterial.emission[0], objMaterial.emission[1], objMaterial.emission[2]);
		material.metalic = 20.0f;
		material.specular = 1.0f;
		this->materials.push_back(material);
	}

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, buffer.size() * sizeof(float), &buffer.at(0), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	
	glBindVertexArray(0);

	bufferSize = buffer.size();


	timer.End("[Mesh] Load Success");
}

Mesh::~Mesh()
{
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
}

#pragma endregion Mesh