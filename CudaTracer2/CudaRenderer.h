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

	float deltaTime = 0;

	// Opengl - Cuda
	GLuint programID;
	GLuint VBO, VAO;

	// Cuda
	GLuint viewGLTexture;
	cudaGraphicsResource* viewResource;
	cudaArray* viewArray;

	// Scene
	std::vector<Mesh*> meshes;
	std::vector<KDTreeGPU*> trees;
	thrust::host_vector<Sphere> spheres;
	thrust::host_vector<Material> materials;

	// GUI
	ViewType viewType = ViewType::ACCU;
	RenderOption currentOption;
	bool uiRenderingWindow = false;
	bool uiFileSaveDialog = false;
};

#endif