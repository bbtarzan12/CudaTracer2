#ifndef H_FRACTALRENDERER
#define H_FRACTALRENDERER

#include "Renderer.h"
#include <thread>

struct cudaGraphicsResource;
struct cudaArray;

class FractalRenderer : public Renderer
{
public:
	FractalRenderer();
	~FractalRenderer() override;

	void Init(RendererOption option) override;
	void SetCamera(CameraOption option) override;
	void Start() override;
	void HandleKeyboard(int key, int scancode, int action, int mods) override;
	void HandleMouseClick(int button, int action, int mods) override;
	void HandleMouseMotion(double xPos, double yPos) override;
	void HandleResize(int width, int height) override;
	void Render(double deltaTime);

private:
	void Update(double deltaTime);

private:
	double deltaTime = 0;

	// Opengl - Cuda
	GLuint cudaViewProgramID;
	GLuint cudaVBO, cudaVAO;

	// Cuda
	GLuint viewGLTexture;
	cudaGraphicsResource* viewResource;
	cudaArray* viewArray;

	// Render Thread
	std::unique_ptr<thread> renderThread;
};

#endif