#include "GLFWManager.h"

#include <iostream>
#include "Renderer.h"

using namespace std;

GLFWManager & GLFWManager::Instance()
{
	static GLFWManager instance;
	return instance;
}

void GLFWManager::Init(int width, int height, const char* name, void* renderer)
{
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
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	GLFWManager::Instance().window = glfwCreateWindow(width, height, name, nullptr, nullptr);
	if (!GLFWManager::Instance().window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(GLFWManager::Instance().window);
	glClear(GL_COLOR_BUFFER_BIT);

	glfwSetWindowUserPointer(GLFWManager::Instance().window, renderer);
	glfwSetInputMode(GLFWManager::Instance().window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetInputMode(GLFWManager::Instance().window, GLFW_STICKY_MOUSE_BUTTONS, GL_TRUE);
	glfwSetKeyCallback(GLFWManager::Instance().window, KeyboardCallback);
	glfwSetMouseButtonCallback(GLFWManager::Instance().window, MouseCallback);
	glfwSetWindowSizeCallback(GLFWManager::Instance().window, ResizeCallback);

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
}

int GLFWManager::WindowShouldClose()
{
	return glfwWindowShouldClose(GLFWManager::Instance().window);
}

GLFWwindow* GLFWManager::GetWindow()
{
	return GLFWManager::Instance().window;
}

void GLFWManager::KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	Renderer* renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	renderer->HandleKeyboard(key, scancode, action, mods);
}

void GLFWManager::ResizeCallback(GLFWwindow* window, int width, int height)
{
	Renderer* renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	renderer->HandleResize(width, height);
}

void GLFWManager::MouseCallback(GLFWwindow * window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_MIDDLE)
	{
		if (action == GLFW_PRESS)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		if (action == GLFW_RELEASE)
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	Renderer* renderer = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	renderer->HandleMouse(button, action, mods);
}

void GLFWManager::ErrorCallback(int errorCode, const char* errorDescription)
{
	fprintf(stderr, "Error: %s\n", errorDescription);
}