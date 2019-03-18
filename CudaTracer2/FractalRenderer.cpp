#include "FractalRenderer.h"
#include "ShaderCommon.h"
#include "FractalKernel.cuh"

using namespace std;

FractalRenderer::FractalRenderer()
{

}

FractalRenderer::~FractalRenderer()
{
	cout << "[Renderer] FractalRenderer Released" << endl;
	glDeleteProgram(cudaViewProgramID);
	glDeleteBuffers(1, &cudaVBO);
	glDeleteVertexArrays(1, &cudaVAO);

	if (renderThread)
		renderThread.release();

	cudaDeviceReset();
}

void FractalRenderer::Init(RendererOption option)
{
	Renderer::Init(option);
	GLFWManager::Init(option.width, option.height, "Fractal Tracer", this);
	cout << "[Renderer] Init FractalRenderer" << endl;
	{
		// Cuda Opengl Interop
		glEnable(GL_TEXTURE_2D);
		glGenTextures(1, &viewGLTexture);
		glBindTexture(GL_TEXTURE_2D, viewGLTexture);
		{
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, option.width, option.height, 0, GL_RGBA, GL_FLOAT, NULL);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		}
		gpuErrorCheck(cudaGraphicsGLRegisterImage(&viewResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		glBindTexture(GL_TEXTURE_2D, 0);

		cudaViewProgramID = LoadShaders("cudaView.vert", "cudaView.frag");

		GLfloat vertices[] =
		{
			// positions        // normal          // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 0.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f,  0.0f, 1.0f,
			1.0f,  1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 1.0f,
		};

		glGenVertexArrays(1, &cudaVAO);
		glGenBuffers(1, &cudaVBO);

		glBindVertexArray(cudaVAO);
		glBindBuffer(GL_ARRAY_BUFFER, cudaVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) 0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*) (6 * sizeof(float)));
		glBindVertexArray(0);
	}
}

void FractalRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	camera->UpdateScreen(option.width, option.height);
	camera->UpdateCamera(0);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void FractalRenderer::Start()
{
	cout << "[Renderer] Start FractalRenderer" << endl;
	renderThread = make_unique<thread>(
		[&]()
	{
		while (true)
		{
			Render(deltaTime);
		}
	});

	double lastTime = glfwGetTime();
	while (GLFWManager::WindowShouldClose() == 0)
	{
		glfwPollEvents();

		double currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		Update(deltaTime);
	}
}

void FractalRenderer::HandleKeyboard(int key, int scancode, int action, int mods)
{
}

void FractalRenderer::HandleMouseClick(int button, int action, int mods)
{
}

void FractalRenderer::HandleMouseMotion(double xPos, double yPos)
{
}

void FractalRenderer::HandleResize(int width, int height)
{
}

void FractalRenderer::Update(double deltaTime)
{

}

void FractalRenderer::Render(double deltaTime)
{

}
