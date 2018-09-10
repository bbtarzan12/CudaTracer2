#ifndef GLFWMANAGER_H
#define GLFWMANAGER_H

#include <windows.h> 
#include <GL/glew.h>
#include <GL/glfw3.h>

class Renderer;

class GLFWManager
{
public:
	static GLFWManager& Instance();

	static void Init(int width, int height, const char* name, void* renderer);

	// GLFW APIs
	static int WindowShouldClose();
	static GLFWwindow* GetWindow();

	// Callbacks
	static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void ResizeCallback(GLFWwindow* window, int width, int height);
	static void MouseCallback(GLFWwindow* window, int button, int action, int mods);
	static void ErrorCallback(int errorCode, const char* errorDescription);

private:
	GLFWwindow * window;

	GLFWManager() = default;
	~GLFWManager() = default;
	GLFWManager(const GLFWManager&) = delete;
	GLFWManager(GLFWManager&&) = delete;
	GLFWManager& operator=(const GLFWManager&) = delete;
	GLFWManager& operator=(GLFWManager&&) = delete;

};

#endif