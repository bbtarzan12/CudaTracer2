#ifndef H_KDTREE
#define H_KDTREE

#include "Mesh.h"

#include <thrust/host_vector.h>
#include <glm.hpp>
#include "cuda_runtime.h"


using namespace std;
using namespace glm;

constexpr float KD_TREE_EPSILON = 0.001f;
constexpr int NUM_TRIS_PER_NODE = 6;

enum SplitAxis
{
	X_AXIS = 0,
	Y_AXIS = 1,
	Z_AXIS = 2
};

enum AABBFace
{
	LEFT = 0,
	FRONT = 1,
	RIGHT = 2,
	BACK = 3,
	TOP = 4,
	BOTTOM = 5
};

struct Triangle
{
	__host__ __device__ Triangle(vec3 pos0 = vec3(0), vec3 pos1 = vec3(0), vec3 pos2 = vec3(0), vec3 nor0 = vec3(0), vec3 nor1 = vec3(0), vec3 nor2 = vec3(0), int materialID = 0);
	vec3 pos[3];
	vec3 nor[3];
	int materialID;
};

struct boundingBox
{
	vec3 min, max;
};

class KDTreeBuilderNode
{
public:
	KDTreeBuilderNode(void);
	~KDTreeBuilderNode(void);

	boundingBox bbox;
	KDTreeBuilderNode *left;
	KDTreeBuilderNode *right;
	int num_tris;
	int *tri_indices;

	SplitAxis split_plane_axis;
	float split_plane_value;

	bool is_leaf_node;

	// One rope for each face of the AABB encompassing the triangles in a node.
	KDTreeBuilderNode *ropes[6];

	int id;

	bool isPointToLeftOfSplittingPlane(const glm::vec3 &p) const;
	KDTreeBuilderNode* getNeighboringNode(glm::vec3 p);

	// Debug method.
	void prettyPrint(void);
};

class KDTreeNode
{
public:
	KDTreeNode(void);

	boundingBox bbox;
	int left_child_index;
	int right_child_index;
	int first_tri_index;
	int num_tris;

	int neighbor_node_indices[6];

	SplitAxis split_plane_axis;
	float split_plane_value;

	bool is_leaf_node;

	bool isPointToLeftOfSplittingPlane(const glm::vec3 &p) const;
	int getNeighboringNodeIndex(glm::vec3 p);

	// Debug method.
	void prettyPrint(void);
};

class KDTreeBuilder
{
public:
	KDTreeBuilder(thrust::host_vector<vec3> verts, thrust::host_vector<ivec3> tris);
	~KDTreeBuilder(void);

	void buildRopeStructure(void);

	// kd-tree getters.
	KDTreeBuilderNode* getRootNode(void) const;
	int getNumLevels(void) const;
	int getNumLeaves(void) const;
	int getNumNodes(void) const;

	// Input mesh getters.
	int getMeshNumVerts(void) const;
	int getMeshNumTris(void) const;
	thrust::host_vector<vec3> getMeshVerts(void) const;
	thrust::host_vector<ivec3> getMeshTris(void) const;

	// Debug methods.
	void printNumTrianglesInEachNode(KDTreeBuilderNode *curr_node, int curr_depth = 1);
	void printNodeIdsAndBounds(KDTreeBuilderNode *curr_node);

private:
	// kd-tree variables.
	KDTreeBuilderNode *root;
	int num_levels, num_leaves, num_nodes;

	thrust::host_vector<vec3> verts;
	thrust::host_vector<ivec3> tris;

	KDTreeBuilderNode* constructTreeMedianSpaceSplit(int num_tris, int *tri_indices, boundingBox bounds, int curr_depth);

	// Rope construction.
	void buildRopeStructure(KDTreeBuilderNode *curr_node, KDTreeBuilderNode *ropes[], bool is_single_ray_case = false);
	void optimizeRopes(KDTreeBuilderNode *ropes[], boundingBox bbox);

	// Bounding box getters.
	SplitAxis getLongestBoundingBoxSide(glm::vec3 min, glm::vec3 max);
	boundingBox computeTightFittingBoundingBox(thrust::host_vector<vec3> verts);
	boundingBox computeTightFittingBoundingBox(int num_tris, int *tri_indices);

	// Triangle getters.
	float getMinTriValue(int tri_index, SplitAxis axis);
	float getMaxTriValue(int tri_index, SplitAxis axis);
};

class KDTree
{
public:
	KDTree(thrust::host_vector<vec3> verts, thrust::host_vector<ivec3> vertexIndices, thrust::host_vector<vec3> norms, thrust::host_vector<ivec3> normalIndices, thrust::host_vector<Material> materials, thrust::host_vector<int> materialIndices);
	~KDTree(void);

	// Getters.
	int getRootIndex(void) const;
	thrust::host_vector<KDTreeNode> getTreeNodes(void) const;
	thrust::host_vector<vec3> getMeshVerts(void) const;
	thrust::host_vector<vec3> getMeshNorms(void) const;
	thrust::host_vector<Material> getMeshMaterials(void) const;
	thrust::host_vector<ivec3> getVertexIndices(void) const;
	thrust::host_vector<ivec3> getNormalIndices(void) const;
	thrust::host_vector<int> getMaterialIndices(void) const;
	thrust::host_vector<int> getTriIndexList(void);
	int getNumNodes(void) const;

	// Debug method.
	void printGPUNodeDataWithCorrespondingCPUNodeData(KDTreeBuilderNode *curr_node, bool pause_on_each_node = false);

private:
	KDTreeBuilder* builder;
	thrust::host_vector<KDTreeNode> tree_nodes;
	thrust::host_vector<int> tri_index_list;

	int num_nodes;
	int root_index;

	// Input mesh variables.
	int num_verts, num_tris;
	thrust::host_vector<vec3> verts;
	thrust::host_vector<vec3> norms;
	thrust::host_vector<Material> materials;
	thrust::host_vector<ivec3> vertexIndices;
	thrust::host_vector<ivec3> normalIndices;
	thrust::host_vector<int> materialIndices;

	void buildTree(KDTreeBuilderNode *curr_node);
};

#endif