#include "Mesh.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

using namespace std;

#pragma region Triangle

__host__ __device__ Triangle::Triangle(vec3 pos0 /*= vec3(0)*/, vec3 pos1 /*= vec3(0)*/, vec3 pos2 /*= vec3(0)*/, vec3 nor0 /*= vec3(0)*/, vec3 nor1 /*= vec3(0)*/, vec3 nor2 /*= vec3(0)*/, int materialID /*= 0*/)
{
	pos[0] = pos0;
	pos[1] = pos1;
	pos[2] = pos2;
	nor[0] = normalize(nor0);
	nor[1] = normalize(nor1);
	nor[2] = normalize(nor2);
	this->materialID = materialID;
}

#pragma endregion Triangle

#pragma region KDTreeNode

////////////////////////////////////////////////////
// KDTreeNode.
////////////////////////////////////////////////////

KDTreeNode::KDTreeNode()
{
	left = nullptr;
	right = nullptr;
	is_leaf_node = false;
	for (auto & rope : ropes)
	{
		rope = nullptr;
	}
	id = -99;
}

KDTreeNode::~KDTreeNode()
{
	if (num_tris > 0)
	{
		delete[] tri_indices;
	}

	if (left)
	{
		delete left;
	}
	if (right)
	{
		delete right;
	}
}

bool KDTreeNode::isPointToLeftOfSplittingPlane(const glm::vec3 &p) const
{
	if (split_plane_axis == X_AXIS)
	{
		return (p.x < split_plane_value);
	}
	else if (split_plane_axis == Y_AXIS)
	{
		return (p.y < split_plane_value);
	}
	else if (split_plane_axis == Z_AXIS)
	{
		return (p.z < split_plane_value);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else
	{
		std::cout << "ERROR: split_plane_axis not set to valid value." << std::endl;
		return false;
	}
}

KDTreeNode* KDTreeNode::getNeighboringNode(glm::vec3 p)
{
	// Check left face.
	if (fabs(p.x - bbox.min.x) < KD_TREE_EPSILON)
	{
		return ropes[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - bbox.max.z) < KD_TREE_EPSILON)
	{
		return ropes[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - bbox.max.x) < KD_TREE_EPSILON)
	{
		return ropes[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - bbox.min.z) < KD_TREE_EPSILON)
	{
		return ropes[BACK];
	}
	// Check top face.
	else if (fabs(p.y - bbox.max.y) < KD_TREE_EPSILON)
	{
		return ropes[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - bbox.min.y) < KD_TREE_EPSILON)
	{
		return ropes[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else
	{
		std::cout << "ERROR: Node neighbor could not be returned." << std::endl;
		return nullptr;
	}
}

void KDTreeNode::prettyPrint()
{
	std::cout << "id: " << id << std::endl;
	std::cout << "bounding box min: ( " << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << " )" << std::endl;
	std::cout << "bounding box max: ( " << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << " )" << std::endl;
	std::cout << "num_tris: " << num_tris << std::endl;

	// Print triangle indices.
	int num_tris_to_print = (num_tris > 10) ? 10 : num_tris;
	for (int i = 0; i < num_tris_to_print; ++i)
	{
		std::cout << "tri index " << i << ": " << tri_indices[i] << std::endl;
	}

	// Print split plane axis.
	if (split_plane_axis == X_AXIS)
	{
		std::cout << "split plane axis: X_AXIS" << std::endl;
	}
	else if (split_plane_axis == Y_AXIS)
	{
		std::cout << "split plane axis: Y_AXIS" << std::endl;
	}
	else if (split_plane_axis == Z_AXIS)
	{
		std::cout << "split plane axis: Z_AXIS" << std::endl;
	}
	else
	{
		std::cout << "split plane axis: invalid" << std::endl;
	}

	std::cout << "split plane value: " << split_plane_value << std::endl;

	// Print whether or not node is a leaf node.
	if (is_leaf_node)
	{
		std::cout << "is leaf node: YES" << std::endl;
	}
	else
	{
		std::cout << "is leaf node: NO" << std::endl;
	}

	// Print pointers to children.
	if (left)
	{
		std::cout << "left child: " << left << std::endl;
	}
	else
	{
		std::cout << "left child: NULL" << std::endl;
	}
	if (right)
	{
		std::cout << "right child: " << right << std::endl;
	}
	else
	{
		std::cout << "right child: NULL" << std::endl;
	}

	// Print neighboring nodes.
	for (int i = 0; i < 6; ++i)
	{
		if (ropes[i])
		{
			std::cout << "rope " << i << ": " << ropes[i] << std::endl;
		}
		else
		{
			std::cout << "rope " << i << ": NULL" << std::endl;
		}
	}

	// Print empty line.
	std::cout << std::endl;
}

#pragma endregion KDTreeNode

#pragma region KDTreeNodeGPU

////////////////////////////////////////////////////
// KDTreeNodeGPU.
////////////////////////////////////////////////////

KDTreeNodeGPU::KDTreeNodeGPU()
{
	left_child_index = -1;
	right_child_index = -1;
	first_tri_index = -1;
	num_tris = 0;
	is_leaf_node = false;

	for (int & neighbor_node_indice : neighbor_node_indices)
	{
		neighbor_node_indice = -1;
	}
}

bool KDTreeNodeGPU::isPointToLeftOfSplittingPlane(const glm::vec3 &p) const
{
	if (split_plane_axis == X_AXIS)
	{
		return (p.x < split_plane_value);
	}
	else if (split_plane_axis == Y_AXIS)
	{
		return (p.y < split_plane_value);
	}
	else if (split_plane_axis == Z_AXIS)
	{
		return (p.z < split_plane_value);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else
	{
		std::cout << "ERROR: split_plane_axis not set to valid value." << std::endl;
		return false;
	}
}

int KDTreeNodeGPU::getNeighboringNodeIndex(glm::vec3 p)
{
	// Check left face.
	if (fabs(p.x - bbox.min.x) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - bbox.max.z) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - bbox.max.x) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - bbox.min.z) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[BACK];
	}
	// Check top face.
	else if (fabs(p.y - bbox.max.y) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - bbox.min.y) < KD_TREE_EPSILON)
	{
		return neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else
	{
		std::cout << "ERROR: Node neighbor could not be returned." << std::endl;
		return -1;
	}
}

void KDTreeNodeGPU::prettyPrint()
{
	std::cout << "bounding box min: ( " << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << " )" << std::endl;
	std::cout << "bounding box max: ( " << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << " )" << std::endl;
	std::cout << "num_tris: " << num_tris << std::endl;
	std::cout << "first_tri_index: " << first_tri_index << std::endl;

	// Print split plane axis.
	if (split_plane_axis == X_AXIS)
	{
		std::cout << "split plane axis: X_AXIS" << std::endl;
	}
	else if (split_plane_axis == Y_AXIS)
	{
		std::cout << "split plane axis: Y_AXIS" << std::endl;
	}
	else if (split_plane_axis == Z_AXIS)
	{
		std::cout << "split plane axis: Z_AXIS" << std::endl;
	}
	else
	{
		std::cout << "split plane axis: invalid" << std::endl;
	}

	std::cout << "split plane value: " << split_plane_value << std::endl;

	// Print whether or not node is a leaf node.
	if (is_leaf_node)
	{
		std::cout << "is leaf node: YES" << std::endl;
	}
	else
	{
		std::cout << "is leaf node: NO" << std::endl;
	}

	// Print children indices.
	std::cout << "left child index: " << left_child_index << std::endl;
	std::cout << "right child index: " << right_child_index << std::endl;

	// Print neighboring nodes.
	for (int i = 0; i < 6; ++i)
	{
		std::cout << "neighbor node index " << i << ": " << neighbor_node_indices[i] << std::endl;
	}

	// Print empty line.
	std::cout << std::endl;
}

#pragma endregion KDTreeNodeGPU

#pragma region KDTreeCPU

////////////////////////////////////////////////////
// Constructor/destructor.
////////////////////////////////////////////////////

KDTreeCPU::KDTreeCPU(int num_tris, glm::vec3 *tris, int num_verts, glm::vec3 *verts)
{
	// Set class-level variables.
	num_levels = 0;
	num_leaves = 0;
	num_nodes = 0;
	this->num_verts = num_verts;
	this->num_tris = num_tris;

	this->verts = new glm::vec3[num_verts];
	for (int i = 0; i < num_verts; ++i)
	{
		this->verts[i] = verts[i];
	}

	this->tris = new glm::vec3[num_tris];
	for (int i = 0; i < num_tris; ++i)
	{
		this->tris[i] = tris[i];
	}

	// Create list of triangle indices for first level of kd-tree.
	int *tri_indices = new int[num_tris];
	for (int i = 0; i < num_tris; ++i)
	{
		tri_indices[i] = i;
	}

	// Compute bounding box for all triangles.
	boundingBox bbox = computeTightFittingBoundingBox(num_verts, this->verts);

	// Build kd-tree and set root node.
	root = constructTreeMedianSpaceSplit(num_tris, tri_indices, bbox, 1);

	// build rope structure
	//KDTreeNode* ropes[6] = { NULL };
	//buildRopeStructure( root, ropes, true );
}

KDTreeCPU::~KDTreeCPU()
{
	if (num_verts > 0)
	{
		delete[] verts;
	}

	if (num_tris > 0)
	{
		delete[] tris;
	}

	delete root;
}


void KDTreeCPU::buildRopeStructure()
{
	KDTreeNode* ropes[6] = { nullptr };
	buildRopeStructure(root, ropes, true);
}


////////////////////////////////////////////////////
// Getters.
////////////////////////////////////////////////////

KDTreeNode* KDTreeCPU::getRootNode() const
{
	return root;
}

int KDTreeCPU::getNumLevels() const
{
	return num_levels;
}

int KDTreeCPU::getNumLeaves(void) const
{
	return num_leaves;
}

int KDTreeCPU::getNumNodes(void) const
{
	return num_nodes;
}

SplitAxis KDTreeCPU::getLongestBoundingBoxSide(glm::vec3 min, glm::vec3 max)
{
	// max > min is guaranteed.
	float xlength = max.x - min.x;
	float ylength = max.y - min.y;
	float zlength = max.z - min.z;
	return (xlength > ylength && xlength > zlength) ? X_AXIS : (ylength > zlength ? Y_AXIS : Z_AXIS);
}

float KDTreeCPU::getMinTriValue(int tri_index, SplitAxis axis)
{
	glm::vec3 tri = tris[tri_index];
	glm::vec3 v0 = verts[(int) tri[0]];
	glm::vec3 v1 = verts[(int) tri[1]];
	glm::vec3 v2 = verts[(int) tri[2]];

	if (axis == X_AXIS)
	{
		return (v0.x < v1.x && v0.x < v2.x) ? v0.x : (v1.x < v2.x ? v1.x : v2.x);
	}
	else if (axis == Y_AXIS)
	{
		return (v0.y < v1.y && v0.y < v2.y) ? v0.y : (v1.y < v2.y ? v1.y : v2.y);
	}
	else
	{
		return (v0.z < v1.z && v0.z < v2.z) ? v0.z : (v1.z < v2.z ? v1.z : v2.z);
	}
}

float KDTreeCPU::getMaxTriValue(int tri_index, SplitAxis axis)
{
	glm::vec3 tri = tris[tri_index];
	glm::vec3 v0 = verts[(int) tri[0]];
	glm::vec3 v1 = verts[(int) tri[1]];
	glm::vec3 v2 = verts[(int) tri[2]];

	if (axis == X_AXIS)
	{
		return (v0.x > v1.x && v0.x > v2.x) ? v0.x : (v1.x > v2.x ? v1.x : v2.x);
	}
	else if (axis == Y_AXIS)
	{
		return (v0.y > v1.y && v0.y > v2.y) ? v0.y : (v1.y > v2.y ? v1.y : v2.y);
	}
	else
	{
		return (v0.z > v1.z && v0.z > v2.z) ? v0.z : (v1.z > v2.z ? v1.z : v2.z);
	}
}

int KDTreeCPU::getMeshNumVerts(void) const
{
	return num_verts;
}

int KDTreeCPU::getMeshNumTris(void) const
{
	return num_tris;
}

glm::vec3* KDTreeCPU::getMeshVerts(void) const
{
	return verts;
}

glm::vec3* KDTreeCPU::getMeshTris(void) const
{
	return tris;
}


////////////////////////////////////////////////////
// Methods to compute tight fitting bounding boxes around triangles.
////////////////////////////////////////////////////

boundingBox KDTreeCPU::computeTightFittingBoundingBox(int num_verts, glm::vec3 *verts)
{
	// Compute bounding box for input mesh.
	glm::vec3 max = glm::vec3(-10000000, -10000000, -10000000);
	glm::vec3 min = glm::vec3(10000000, 10000000, 10000000);

	for (int i = 0; i < num_verts; ++i)
	{
		if (verts[i].x < min.x)
		{
			min.x = verts[i].x;
		}
		if (verts[i].y < min.y)
		{
			min.y = verts[i].y;
		}
		if (verts[i].z < min.z)
		{
			min.z = verts[i].z;
		}
		if (verts[i].x > max.x)
		{
			max.x = verts[i].x;
		}
		if (verts[i].y > max.y)
		{
			max.y = verts[i].y;
		}
		if (verts[i].z > max.z)
		{
			max.z = verts[i].z;
		}
	}

	boundingBox bbox;
	bbox.min = min;
	bbox.max = max;

	return bbox;
}

boundingBox KDTreeCPU::computeTightFittingBoundingBox(int num_tris, int *tri_indices)
{
	int num_verts = num_tris * 3;
	glm::vec3 *verts = new glm::vec3[num_verts];

	int verts_index;
	for (int i = 0; i < num_tris; ++i)
	{
		glm::vec3 tri = tris[i];
		verts_index = i * 3;
		verts[verts_index + 0] = this->verts[(int) tri[0]];
		verts[verts_index + 1] = this->verts[(int) tri[1]];
		verts[verts_index + 2] = this->verts[(int) tri[2]];
	}

	boundingBox bbox = computeTightFittingBoundingBox(num_verts, verts);
	delete[] verts;
	return bbox;
}


////////////////////////////////////////////////////
// constructTreeMedianSpaceSplit().
////////////////////////////////////////////////////
KDTreeNode* KDTreeCPU::constructTreeMedianSpaceSplit(int num_tris, int *tri_indices, boundingBox bounds, int curr_depth)
{
	// Create new node.
	KDTreeNode *node = new KDTreeNode();
	node->num_tris = num_tris;
	node->tri_indices = tri_indices;

	// Override passed-in bounding box and create "tightest-fitting" bounding box around passed-in list of triangles.
	if (USE_TIGHT_FITTING_BOUNDING_BOXES)
	{
		node->bbox = computeTightFittingBoundingBox(num_tris, tri_indices);
	}
	else
	{
		node->bbox = bounds;
	}

	// Base case--Number of triangles in node is small enough.
	if (num_tris <= NUM_TRIS_PER_NODE || curr_depth > 20)
	{
		node->is_leaf_node = true;

		// Update number of tree levels.
		if (curr_depth > num_levels)
		{
			num_levels = curr_depth;
		}

		// Set node ID.
		node->id = num_nodes;
		++num_nodes;

		// Return leaf node.
		++num_leaves;
		return node;
	}

	// Get longest side of bounding box.
	SplitAxis longest_side = getLongestBoundingBoxSide(bounds.min, bounds.max);
	node->split_plane_axis = longest_side;

	// Compute median value for longest side as well as "loose-fitting" bounding boxes.
	float median_val = 0.0;
	boundingBox left_bbox = node->bbox;
	boundingBox right_bbox = node->bbox;
	if (longest_side == X_AXIS)
	{
		median_val = bounds.min.x + ((bounds.max.x - bounds.min.x) / 2.0f);
		left_bbox.max.x = median_val;
		right_bbox.min.x = median_val;
	}
	else if (longest_side == Y_AXIS)
	{
		median_val = bounds.min.y + ((bounds.max.y - bounds.min.y) / 2.0f);
		left_bbox.max.y = median_val;
		right_bbox.min.y = median_val;
	}
	else
	{
		median_val = bounds.min.z + ((bounds.max.z - bounds.min.z) / 2.0f);
		left_bbox.max.z = median_val;
		right_bbox.min.z = median_val;
	}

	node->split_plane_value = median_val;

	// Allocate and initialize memory for temporary buffers to hold triangle indices for left and right subtrees.
	int *temp_left_tri_indices = new int[num_tris];
	int *temp_right_tri_indices = new int[num_tris];

	// Populate temporary buffers.
	int left_tri_count = 0, right_tri_count = 0;
	float min_tri_val, max_tri_val;
	for (int i = 0; i < num_tris; ++i)
	{
		// Get min and max triangle values along desired axis.
		if (longest_side == X_AXIS)
		{
			min_tri_val = getMinTriValue(tri_indices[i], X_AXIS);
			max_tri_val = getMaxTriValue(tri_indices[i], X_AXIS);
		}
		else if (longest_side == Y_AXIS)
		{
			min_tri_val = getMinTriValue(tri_indices[i], Y_AXIS);
			max_tri_val = getMaxTriValue(tri_indices[i], Y_AXIS);
		}
		else
		{
			min_tri_val = getMinTriValue(tri_indices[i], Z_AXIS);
			max_tri_val = getMaxTriValue(tri_indices[i], Z_AXIS);
		}

		// Update temp_left_tri_indices.
		if (min_tri_val < median_val)
		{
			temp_left_tri_indices[i] = tri_indices[i];
			++left_tri_count;
		}
		else
		{
			temp_left_tri_indices[i] = -1;
		}

		// Update temp_right_tri_indices.
		if (max_tri_val >= median_val)
		{
			temp_right_tri_indices[i] = tri_indices[i];
			++right_tri_count;
		}
		else
		{
			temp_right_tri_indices[i] = -1;
		}
	}

	// Allocate memory for lists of triangle indices for left and right subtrees.
	int *left_tri_indices = new int[left_tri_count];
	int *right_tri_indices = new int[right_tri_count];

	// Populate lists of triangle indices.
	int left_index = 0, right_index = 0;
	for (int i = 0; i < num_tris; ++i)
	{
		if (temp_left_tri_indices[i] != -1)
		{
			left_tri_indices[left_index] = temp_left_tri_indices[i];
			++left_index;
		}
		if (temp_right_tri_indices[i] != -1)
		{
			right_tri_indices[right_index] = temp_right_tri_indices[i];
			++right_index;
		}
	}

	// Free temporary triangle indices buffers.
	delete[] temp_left_tri_indices;
	delete[] temp_right_tri_indices;

	// Recurse.
	node->left = constructTreeMedianSpaceSplit(left_tri_count, left_tri_indices, left_bbox, curr_depth + 1);
	node->right = constructTreeMedianSpaceSplit(right_tri_count, right_tri_indices, right_bbox, curr_depth + 1);

	// Set node ID.
	node->id = num_nodes;
	++num_nodes;

	return node;
}

////////////////////////////////////////////////////
// Connect kd-tree nodes with ropes.
// Tree construction post-process.
////////////////////////////////////////////////////
void KDTreeCPU::buildRopeStructure(KDTreeNode *curr_node, KDTreeNode *ropes[], bool is_single_ray_case)
{
	// Base case.
	if (curr_node->is_leaf_node)
	{
		//std::cout<<curr_node->id<<": "<<std::endl;
		for (int i = 0; i < 6; ++i)
		{
			curr_node->ropes[i] = ropes[i];
		}
	}
	else
	{
		// Only optimize ropes on single ray case.
		// It is not optimal to optimize on packet traversal case.
		if (is_single_ray_case)
		{
			optimizeRopes(ropes, curr_node->bbox);
		}

		AABBFace SL, SR;
		if (curr_node->split_plane_axis == X_AXIS)
		{
			SL = LEFT;
			SR = RIGHT;
		}
		else if (curr_node->split_plane_axis == Y_AXIS)
		{
			SL = BOTTOM;
			SR = TOP;
		}
		// Split plane is Z_AXIS.
		else
		{
			SL = BACK;
			SR = FRONT;
		}

		KDTreeNode* RS_left[6];
		KDTreeNode* RS_right[6];
		for (int i = 0; i < 6; ++i)
		{
			RS_left[i] = ropes[i];
			RS_right[i] = ropes[i];
		}

		// Recurse.
		RS_left[SR] = curr_node->right;
		buildRopeStructure(curr_node->left, RS_left, is_single_ray_case);

		// Recurse.
		RS_right[SL] = curr_node->left;
		buildRopeStructure(curr_node->right, RS_right, is_single_ray_case);
	}
}


////////////////////////////////////////////////////
// Optimization step called in certain cases when constructing stackless kd-tree rope structure.
////////////////////////////////////////////////////
void KDTreeCPU::optimizeRopes(KDTreeNode *ropes[], boundingBox bbox)
{
	// Loop through ropes of all faces of node bounding box.
	for (int i = 0; i < 6; ++i)
	{
		KDTreeNode *rope_node = ropes[i];

		if (rope_node == nullptr)
		{
			continue;
		}

		// Process until leaf node is reached.
		// The optimization - We try to push the ropes down into the tree as far as possible
		// instead of just having the ropes point to the roots of neighboring subtrees.
		while (!rope_node->is_leaf_node)
		{

			// Case I.

			if (i == LEFT || i == RIGHT)
			{

				// Case I-A.

				// Handle parallel split plane case.
				if (rope_node->split_plane_axis == X_AXIS)
				{
					rope_node = (i == LEFT) ? rope_node->right : rope_node->left;
				}

				// Case I-B.

				else if (rope_node->split_plane_axis == Y_AXIS)
				{
					if (rope_node->split_plane_value < (bbox.min.y - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.y + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}

				// Case I-C.

				// Split plane is Z_AXIS.
				else
				{
					if (rope_node->split_plane_value < (bbox.min.z - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.z + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}
			}

			// Case II.

			else if (i == FRONT || i == BACK)
			{

				// Case II-A.

				// Handle parallel split plane case.
				if (rope_node->split_plane_axis == Z_AXIS)
				{
					rope_node = (i == BACK) ? rope_node->right : rope_node->left;
				}

				// Case II-B.

				else if (rope_node->split_plane_axis == X_AXIS)
				{
					if (rope_node->split_plane_value < (bbox.min.x - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.x + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}

				// Case II-C.

				// Split plane is Y_AXIS.
				else
				{
					if (rope_node->split_plane_value < (bbox.min.y - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.y + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}
			}

			// Case III.

			// TOP and BOTTOM.
			else
			{

				// Case III-A.

				// Handle parallel split plane case.
				if (rope_node->split_plane_axis == Y_AXIS)
				{
					rope_node = (i == BOTTOM) ? rope_node->right : rope_node->left;
				}

				// Case III-B.

				else if (rope_node->split_plane_axis == Z_AXIS)
				{
					if (rope_node->split_plane_value < (bbox.min.z - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.z + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}

				// Case III-C.

				// Split plane is X_AXIS.
				else
				{
					if (rope_node->split_plane_value < (bbox.min.x - KD_TREE_EPSILON))
					{
						rope_node = rope_node->right;
					}
					else if (rope_node->split_plane_value > (bbox.max.x + KD_TREE_EPSILON))
					{
						rope_node = rope_node->left;
					}
					else
					{
						break;
					}
				}
			}
		}
	}
}


////////////////////////////////////////////////////
// Debug methods.
////////////////////////////////////////////////////

void KDTreeCPU::printNumTrianglesInEachNode(KDTreeNode *curr_node, int curr_depth)
{
	std::cout << "Level: " << curr_depth << ", Triangles: " << curr_node->num_tris << std::endl;

	if (curr_node->left)
	{
		printNumTrianglesInEachNode(curr_node->left, curr_depth + 1);
	}
	if (curr_node->right)
	{
		printNumTrianglesInEachNode(curr_node->right, curr_depth + 1);
	}
}

void KDTreeCPU::printNodeIdsAndBounds(KDTreeNode *curr_node)
{
	std::cout << "Node ID: " << curr_node->id << std::endl;
	std::cout << "Node bbox min: ( " << curr_node->bbox.min.x << ", " << curr_node->bbox.min.y << ", " << curr_node->bbox.min.z << " )" << std::endl;
	std::cout << "Node bbox max: ( " << curr_node->bbox.max.x << ", " << curr_node->bbox.max.y << ", " << curr_node->bbox.max.z << " )" << std::endl;
	std::cout << std::endl;

	if (curr_node->left)
	{
		printNodeIdsAndBounds(curr_node->left);
	}
	if (curr_node->right)
	{
		printNodeIdsAndBounds(curr_node->right);
	}
}

#pragma endregion KDTreeCPU

#pragma region KDTreeGPU


////////////////////////////////////////////////////
// Construtor/destructor.
////////////////////////////////////////////////////

KDTreeGPU::KDTreeGPU(KDTreeCPU *kd_tree_cpu)
{
	num_nodes = kd_tree_cpu->getNumNodes();
	root_index = kd_tree_cpu->getRootNode()->id;

	num_verts = kd_tree_cpu->getMeshNumVerts();
	num_tris = kd_tree_cpu->getMeshNumTris();

	glm::vec3 *tmp_verts = kd_tree_cpu->getMeshVerts();
	verts = new glm::vec3[num_verts];
	for (int i = 0; i < num_verts; ++i)
	{
		verts[i] = tmp_verts[i];
	}

	glm::vec3 *tmp_tris = kd_tree_cpu->getMeshTris();
	tris = new glm::vec3[num_tris];
	for (int i = 0; i < num_tris; ++i)
	{
		tris[i] = tmp_tris[i];
	}

	// Allocate memory for all nodes in GPU kd-tree.
	tree_nodes = new KDTreeNodeGPU[num_nodes];

	// Populate tree_nodes and tri_index_list.
	tri_index_list.clear();
	buildTree(kd_tree_cpu->getRootNode());
}

KDTreeGPU::~KDTreeGPU()
{
	if (num_verts > 0)
	{
		delete[] verts;
	}

	if (num_tris > 0)
	{
		delete[] tris;
	}

	delete[] tree_nodes;
}


////////////////////////////////////////////////////
// Getters.
////////////////////////////////////////////////////

int KDTreeGPU::getRootIndex() const
{
	return root_index;
}

KDTreeNodeGPU* KDTreeGPU::getTreeNodes() const
{
	return tree_nodes;
}

glm::vec3* KDTreeGPU::getMeshVerts() const
{
	return verts;
}

glm::vec3* KDTreeGPU::getMeshTris() const
{
	return tris;
}

std::vector<int> KDTreeGPU::getTriIndexList() const
{
	return tri_index_list;
}

int KDTreeGPU::getNumNodes() const
{
	return num_nodes;
}


////////////////////////////////////////////////////
// Recursive method to build up GPU kd-tree structure from CPU kd-tree structure.
// This method populates tree_nodes, an array of KDTreeNodeGPUs and
// tri_index_list, a list of triangle indices for all leaf nodes to be sent to the device.
////////////////////////////////////////////////////
void KDTreeGPU::buildTree(KDTreeNode *curr_node)
{
	// Get index of node in CPU kd-tree.
	int index = curr_node->id;

	// Start building GPU kd-tree node from current CPU kd-tree node.
	tree_nodes[index].bbox = curr_node->bbox;
	tree_nodes[index].split_plane_axis = curr_node->split_plane_axis;
	tree_nodes[index].split_plane_value = curr_node->split_plane_value;
	tree_nodes[index].is_leaf_node = curr_node->is_leaf_node;

	// Leaf node.
	if (curr_node->is_leaf_node)
	{
		tree_nodes[index].num_tris = curr_node->num_tris;
		tree_nodes[index].first_tri_index = tri_index_list.size(); // tri_index_list initially contains 0 elements.

		// Add triangles to tri_index_list as each leaf node is processed.
		for (int i = 0; i < curr_node->num_tris; ++i)
		{
			tri_index_list.push_back(curr_node->tri_indices[i]);
		}

		// Set neighboring node indices for GPU node using ropes in CPU node.
		for (int i = 0; i < 6; ++i)
		{
			if (curr_node->ropes[i])
			{
				tree_nodes[index].neighbor_node_indices[i] = curr_node->ropes[i]->id;
			}
		}
	}
	else
	{
		if (curr_node->left)
		{
			// Set child node index for current node and recurse.
			tree_nodes[index].left_child_index = curr_node->left->id;
			buildTree(curr_node->left);
		}
		if (curr_node->right)
		{
			// Set child node index for current node and recurse.
			tree_nodes[index].right_child_index = curr_node->right->id;
			buildTree(curr_node->right);
		}
	}
}


////////////////////////////////////////////////////
// Debug methods.
////////////////////////////////////////////////////

void KDTreeGPU::printGPUNodeDataWithCorrespondingCPUNodeData(KDTreeNode *curr_node, bool pause_on_each_node)
{
	curr_node->prettyPrint();
	tree_nodes[curr_node->id].prettyPrint();

	if (pause_on_each_node)
	{
		std::cin.ignore();
	}

	if (curr_node->left)
	{
		printGPUNodeDataWithCorrespondingCPUNodeData(curr_node->left, pause_on_each_node);
	}
	if (curr_node->right)
	{
		printGPUNodeDataWithCorrespondingCPUNodeData(curr_node->right, pause_on_each_node);
	}
}

#pragma endregion KDTreeGPU

#pragma region Mesh

std::istream& safeGetline(std::istream& is, std::string& t)
{
	t.clear();

	// The characters in the stream are read one-by-one using a std::streambuf.
	// That is faster than reading them one-by-one using the std::istream.
	// Code that uses streambuf this way must be guarded by a sentry object.
	// The sentry object performs various tasks,
	// such as thread synchronization and updating the stream state.

	std::istream::sentry se(is, true);
	std::streambuf* sb = is.rdbuf();

	for (;;)
	{
		int c = sb->sbumpc();
		switch (c)
		{
			case '\n':
				return is;
			case '\r':
				if (sb->sgetc() == '\n')
					sb->sbumpc();
				return is;
			case EOF:
				// Also handle the case when the last line has no line ending
				if (t.empty())
					is.setstate(std::ios::eofbit);
				return is;
			default:
				t += (char) c;
		}
	}
}

std::vector<std::string> tokenizeString(std::string str)
{
	std::stringstream strstr(str);
	std::istream_iterator<std::string> it(strstr);
	std::istream_iterator<std::string> end;
	std::vector<std::string> results(it, end);
	return results;
}

__host__ __device__ Mesh::Mesh(vec3 position /*= vec3(0)*/, string fileName /*= ""*/, int materialID /*= 0*/)
{
	this->position = position;

	ifstream ifile;
	string line;
	ifile.open(fileName.c_str());

	vector<glm::vec3> verts, faces;

	while (safeGetline(ifile, line))
	{
		vector<string> tokens = tokenizeString(line);

		if (!tokens.empty() && strcmp(tokens[0].c_str(), "v") == 0)
		{
			verts.emplace_back(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (!tokens.empty() && strcmp(tokens[0].c_str(), "f") == 0)
		{
			char* findex1 = strtok(const_cast<char*>(tokens[1].c_str()), "/");
			char* findex2 = strtok(const_cast<char*>(tokens[2].c_str()), "/");
			char* findex3 = strtok(const_cast<char*>(tokens[3].c_str()), "/");
			faces.emplace_back(atof(findex1) - 1, atof(findex2) - 1, atof(findex3) - 1);
		}
	}

	glm::vec3* vertData = new glm::vec3[verts.size()];
	glm::vec3* faceData = new glm::vec3[faces.size()];

	glm::vec3 max = glm::vec3(-10000, -10000, -10000);
	glm::vec3 min = glm::vec3(10000, 10000, 10000);

	for (int i = 0; i < verts.size(); i++)
	{
		vertData[i] = verts[i];
		if (verts[i].x < min.x)
		{
			min.x = verts[i].x;
		}
		if (verts[i].y < min.y)
		{
			min.y = verts[i].y;
		}
		if (verts[i].z < min.z)
		{
			min.z = verts[i].z;
		}
		if (verts[i].x > max.x)
		{
			max.x = verts[i].x;
		}
		if (verts[i].y > max.y)
		{
			max.y = verts[i].y;
		}
		if (verts[i].z > max.z)
		{
			max.z = verts[i].z;
		}
	}
	for (int i = 0; i < faces.size(); i++)
	{
		faceData[i] = faces[i];
	}

	this->tris = faceData;
	this->verts = vertData;
	this->numTris = faces.size();
	this->numVerts = verts.size();

	bb.min = min;
	bb.max = max;
}

#pragma endregion Mesh