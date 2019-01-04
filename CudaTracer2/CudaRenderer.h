#ifndef H_CUDARENDERER
#define H_CUDARENDERER

#include "Renderer.h"
#include "PathKernel.cuh"

enum class ViewType { OPENGL, ACCU, IMAGE };

class CudaRenderer : public Renderer
{
public:
	CudaRenderer();
	~CudaRenderer() override;

	void Init(RendererOption option) override;
	void SetCamera(CameraOption option) override;
	void Start() override;
	void HandleKeyboard(int key, int scancode, int action, int mods) override;
	void HandleMouseClick(int button, int action, int mods) override;
	void HandleMouseMotion(double xPos, double yPos) override;
	void HandleResize(int width, int height) override;

private:
	void Update(float deltaTime);
	void Render(float deltaTime);
	void LoadObj(const char* fileName);
	void GenerateTree();

	// UI
	void RenderUIMenuBar();
	void RenderMeshListWindow();

	float deltaTime = 0;

	// Opengl 나중에 정리할 것
	GLuint openglViewProgramID;
	GLuint sunDirID, camPosID;
	GLuint metalicID, sunPowerID;
	GLuint modelID, viewID, projectionID;

	// Opengl - Cuda
	GLuint cudaViewProgramID;
	GLuint cudaVBO, cudaVAO;

	// Cuda
	GLuint viewGLTexture;
	cudaGraphicsResource* viewResource;
	cudaArray* viewArray;

	// Scene
	std::vector<shared_ptr<Mesh>> meshes;
	bool rebuildTree;
	KDTree* tree = nullptr;
	thrust::host_vector<Sphere> spheres;

	// Sun
	vec3 sunDirection;
	float sunPitch = 45;
	float sunYaw = 0;
	float sunLuminance = 0.6f;
	float sunExtent = 0.01f;
	bool sunDirty = true;

	// Test Material
	float metalic = 0.f;
	float specular = 0.f;

	// GUI
	ViewType viewType = ViewType::OPENGL;
	RenderOption currentOption;
	bool uiRenderingWindow = false;
	bool uiFileSaveDialog = false;
	bool uiObjLoadDialog = false;
	bool uiMeshWindow = false;
};

#endif