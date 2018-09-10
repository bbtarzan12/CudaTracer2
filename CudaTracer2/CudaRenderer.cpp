#include "CudaRenderer.h"
#include <iostream>

using namespace std;

CudaRenderer::CudaRenderer()
{
}


CudaRenderer::~CudaRenderer()
{
	cout << "[Renderer] CudaRenderer Free" << endl;
}

void CudaRenderer::Init(RendererOption option)
{
	Renderer::Init(option);

	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
	{
		cerr << "[Error] GLFW 초기화 실패" << endl;
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	window = glfwCreateWindow(option.widht, option.height, "Cuda Tracer", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glClear(GL_COLOR_BUFFER_BIT);

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GL_TRUE);
	glfwSetKeyCallback(window, KeyboardCallback);
	glfwSetMouseButtonCallback(window, MouseCallback);
	glfwSetWindowSizeCallback(window, ResizeCallback);

	glewExperimental = GL_TRUE;

	GLenum errorCode = glewInit();
	if (GLEW_OK != errorCode)
	{
		cerr << "[Error] GLEW 초기화 실패" << glewGetErrorString(errorCode) << endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	if (!GLEW_VERSION_3_3)
	{
		cerr << "[Error] 3.3 API가 유효하지 않습니다" << endl;
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	cout << "[OpenGL] OpenGL version: " << glGetString(GL_VERSION) << endl;
	cout << "[OpenGL] GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	cout << "[OpenGL] Vendor: " << glGetString(GL_VENDOR) << endl;
	cout << "[OpenGL] Renderer: " << glGetString(GL_RENDERER) << endl;

	glfwSwapInterval(1); // vSync
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	cout << "[Renderer] CudaRenderer Init" << endl;
}

void CudaRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void CudaRenderer::Start()
{
	cout << "[Renderer] Start CudaRenderer" << endl;

	float lastTime = glfwGetTime();
	while (glfwWindowShouldClose(window) == 0)
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;


		glClear(GL_COLOR_BUFFER_BIT);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void CudaRenderer::KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

void CudaRenderer::ResizeCallback(GLFWwindow* window, int width, int height)
{

}

void CudaRenderer::MouseCallback(GLFWwindow * window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (action == GLFW_PRESS)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		if (action == GLFW_RELEASE)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}

void CudaRenderer::ErrorCallback(int errorCode, const char* errorDescription)
{
	fprintf(stderr, "Error: %s\n", errorDescription);
}