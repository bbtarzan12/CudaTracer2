#ifndef H_GLFWMANAGER
#define H_GLFWMANAGER

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
	static void GetCursorPos(double* x, double* y);
	static void SetCursorToPos(double x, double y);

	// Inputs
	static bool IsKeyDown(int key);
	static bool IsMouseDown(int button);

	// Callbacks
	static void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void ResizeCallback(GLFWwindow* window, int width, int height);
	static void MouseCallback(GLFWwindow* window, int button, int action, int mods);
	static void MousePosCallback(GLFWwindow* window, double xPos, double yPos);
	static void ErrorCallback(int errorCode, const char* errorDescription);

private:
	GLFWwindow * window;
	bool keyState[256] = { false };
	bool mouseState[3] = { false };

	GLFWManager() = default;
	~GLFWManager() = default;
	GLFWManager(const GLFWManager&) = delete;
	GLFWManager(GLFWManager&&) = delete;
	GLFWManager& operator=(const GLFWManager&) = delete;
	GLFWManager& operator=(GLFWManager&&) = delete;

};

#endif