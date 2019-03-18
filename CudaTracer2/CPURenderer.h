#ifndef H_CPURENDERER
#define H_CPURENDERER

#include "Renderer.h"
#include "CPUKernel.h"

#include <thread>
#include <vector>

using namespace CPUKernel;

class CPURenderer : public Renderer
{
public:
	CPURenderer();
	~CPURenderer() override;

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

	// OpenGL
	GLuint viewGLTexture;
	GLuint cudaViewProgramID;
	GLuint cudaVBO, cudaVAO;

	// Scene
	std::vector<Sphere> spheres;
	std::vector<Light*> lights;

	// Render
	RenderOption currentOption;
};

#endif