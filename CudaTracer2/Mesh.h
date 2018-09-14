#ifndef H_MESH
#define H_MESH

#include <glm.hpp>
#include "cuda_runtime.h"
#include <vector>

using namespace glm;

constexpr float KD_TREE_EPSILON = 0.00001f;
constexpr int NUM_TRIS_PER_NODE = 6;
constexpr bool USE_TIGHT_FITTING_BOUNDING_BOXES = true;

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
	glm::vec3 min, max;
};

class KDTreeNode
{
public:
	KDTreeNode(void);
	~KDTreeNode(void);

	boundingBox bbox;
	KDTreeNode *left;
	KDTreeNode *right;
	int num_tris;
	int *tri_indices;

	SplitAxis split_plane_axis;
	float split_plane_value;

	bool is_leaf_node;

	// One rope for each face of the AABB encompassing the triangles in a node.
	KDTreeNode *ropes[6];

	int id;

	bool isPointToLeftOfSplittingPlane(const glm::vec3 &p) const;
	KDTreeNode* getNeighboringNode(glm::vec3 p);

	// Debug method.
	void prettyPrint(void);
};

class KDTreeNodeGPU
{
public:
	KDTreeNodeGPU(void);

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

class KDTreeCPU
{
public:
	KDTreeCPU(int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts);
	~KDTreeCPU(void);

	void buildRopeStructure(void);

	// kd-tree getters.
	KDTreeNode* getRootNode(void) const;
	int getNumLevels(void) const;
	int getNumLeaves(void) const;
	int getNumNodes(void) const;

	// Input mesh getters.
	int getMeshNumVerts(void) const;
	int getMeshNumTris(void) const;
	glm::vec3* getMeshVerts(void) const;
	glm::vec3* getMeshTris(void) const;

	// Debug methods.
	void printNumTrianglesInEachNode(KDTreeNode *curr_node, int curr_depth = 1);
	void printNodeIdsAndBounds(KDTreeNode *curr_node);

private:
	// kd-tree variables.
	KDTreeNode *root;
	int num_levels, num_leaves, num_nodes;

	// Input mesh variables.
	int num_verts, num_tris;
	glm::vec3 *verts, *tris;

	KDTreeNode* constructTreeMedianSpaceSplit(int num_tris, int *tri_indices, boundingBox bounds, int curr_depth);

	// Rope construction.
	void buildRopeStructure(KDTreeNode *curr_node, KDTreeNode *ropes[], bool is_single_ray_case = false);
	void optimizeRopes(KDTreeNode *ropes[], boundingBox bbox);

	// Bounding box getters.
	SplitAxis getLongestBoundingBoxSide(glm::vec3 min, glm::vec3 max);
	boundingBox computeTightFittingBoundingBox(int num_verts, glm::vec3 *verts);
	boundingBox computeTightFittingBoundingBox(int num_tris, int *tri_indices);

	// Triangle getters.
	float getMinTriValue(int tri_index, SplitAxis axis);
	float getMaxTriValue(int tri_index, SplitAxis axis);
};

class KDTreeGPU
{
public:
	KDTreeGPU(KDTreeCPU *kd_tree_cpu);
	~KDTreeGPU(void);

	// Getters.
	int getRootIndex(void) const;
	KDTreeNodeGPU* getTreeNodes(void) const;
	glm::vec3* getMeshVerts(void) const;
	glm::vec3* getMeshTris(void) const;
	std::vector<int> getTriIndexList(void) const;
	int getNumNodes(void) const;

	// Debug method.
	void printGPUNodeDataWithCorrespondingCPUNodeData(KDTreeNode *curr_node, bool pause_on_each_node = false);

private:
	KDTreeNodeGPU *tree_nodes;
	std::vector<int> tri_index_list;

	int num_nodes;
	int root_index;

	// Input mesh variables.
	int num_verts, num_tris;
	glm::vec3 *verts, *tris;

	void buildTree(KDTreeNode *curr_node);
};

class Mesh
{
public:
	__host__ __device__ Mesh(vec3 position = vec3(0), std::string fileName = "", int materialID = 0);

	vec3 position;
	int numTris, numVerts;
	glm::vec3* tris;
	glm::vec3* verts;
	boundingBox bb;
};

#endif