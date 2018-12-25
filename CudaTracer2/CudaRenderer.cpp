#include "CudaRenderer.h"
#include "ShaderCommon.h"
#include "PathKernel.cuh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

#include <FreeImage.h>

#include <iostream>
#include <algorithm>

using namespace std;

const char* viewTypeArray[] = { "OpenGL", "Accumulate", "Image" };

CudaRenderer::CudaRenderer()
{
}


CudaRenderer::~CudaRenderer()
{
	cout << "[Renderer] CudaRenderer Free" << endl;
	glDeleteProgram(cudaViewProgramID);
	glDeleteBuffers(1, &cudaVBO);
	glDeleteVertexArrays(1, &cudaVAO);
	cudaDeviceReset();
}

void CudaRenderer::Init(RendererOption option)
{
	Renderer::Init(option);
	GLFWManager::Init(option.width, option.height, "Cuda Tracer", this);
	InitHDRTexture("river_walk_1_4k.hdr");
	cout << "[Renderer] CudaRenderer Init" << endl;

	{
		// Scene
		//spheres.push_back(Sphere(vec3(0, 1140, 0), 1000, 1));
		//spheres.push_back(Sphere(vec3(0, -1140, 0), 1000, 4));
		//spheres.push_back(Sphere(vec3(1140, 0, 0), 1000, 4));
		//spheres.push_back(Sphere(vec3(-1140, 0, 0), 1000, 4));
		//spheres.push_back(Sphere(vec3(0, 0, 1140), 1000, 6));
		//spheres.push_back(Sphere(vec3(0, 0, -1140), 1000, 5));
		
		// Gen Opengl Buffers and Shaders
		//LoadObj("test3.obj");
	}

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

	{
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImGuiIO& io = ImGui::GetIO(); (void) io;
		io.WantCaptureKeyboard = true;
		io.WantCaptureMouse = true;
		ImGui_ImplGlfw_InitForOpenGL(GLFWManager::GetWindow(), false);
		ImGui_ImplOpenGL3_Init("#version 130");
		ImGui::StyleColorsDark();
	}
}

void CudaRenderer::SetCamera(CameraOption option)
{
	Renderer::SetCamera(option);
	camera->UpdateScreen(option.width, option.height);
	camera->UpdateCamera(0);
	cout << "[Renderer] Set Camera To Renderer" << endl;
}

void CudaRenderer::Start()
{
	cout << "[Renderer] Start CudaRenderer" << endl;

	float lastTime = glfwGetTime();
	while (GLFWManager::WindowShouldClose() == 0)
	{
		glfwPollEvents();

		float currentTime = glfwGetTime();
		deltaTime = currentTime - lastTime;
		lastTime = currentTime;

		Update(deltaTime);
		Render(deltaTime);
	}
}

void CudaRenderer::Update(float deltaTime)
{
	vec2 input = vec2(GLFWManager::IsKeyDown(GLFW_KEY_W) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_S) ? -1 : 0, GLFWManager::IsKeyDown(GLFW_KEY_D) ? 1 : GLFWManager::IsKeyDown(GLFW_KEY_A) ? -1 : 0);
	camera->UpdateCamera(deltaTime, input);
}

void CudaRenderer::Render(float deltaTime)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	const int width = camera->width;
	const int height = camera->height;


	// OpenGL + Cuda
	switch (viewType)
	{
		case ViewType::OPENGL:

			break;
		case ViewType::ACCU:
			gpuErrorCheck(cudaGraphicsMapResources(1, &viewResource));
			gpuErrorCheck(cudaGraphicsSubResourceGetMappedArray(&viewArray, viewResource, 0, 0));

			cudaResourceDesc viewCudaArrayResourceDesc;
			{
				viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
				viewCudaArrayResourceDesc.res.array.array = viewArray;
			}

			cudaSurfaceObject_t viewCudaSurfaceObject;
			gpuErrorCheck(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
			{
				if (camera->dirty)
				{
					currentOption.frame = 1;
					currentOption.isAccumulate = true;
					currentOption.surf = viewCudaSurfaceObject;
					camera->ResetDirty();
				}

				RenderKernel(camera, spheres, tree, currentOption);
			}
			gpuErrorCheck(cudaDestroySurfaceObject(viewCudaSurfaceObject));
			gpuErrorCheck(cudaGraphicsUnmapResources(1, &viewResource));
			cudaStreamSynchronize(0);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, viewGLTexture);

			glUseProgram(cudaViewProgramID);

			glBindVertexArray(cudaVAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			glBindVertexArray(0);
			currentOption.frame++;
			break;
		case ViewType::IMAGE:
		{
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, viewGLTexture);
			glUseProgram(cudaViewProgramID);
			glBindVertexArray(cudaVAO);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
			glBindVertexArray(0);
		}
		{
			ImGui::SetNextWindowPos(ImVec2(3, 23));
			ImGui::Begin("Image", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove);

			ImGui::Text("Samples : %d", currentOption.frame);
			if (ImGui::Button("Save Image"))
			{
				uiFileSaveDialog = true;
			}

			ImGui::End();
		}
		break;
		default:
			break;
	}

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				glfwSetWindowShouldClose(GLFWManager::GetWindow(), GLFW_TRUE);
			}
			if (ImGui::MenuItem("Import Obj")) { uiObjLoadDialog = true; }
			ImGui::EndMenu();
		}
		ImGui::Separator();
		if (ImGui::BeginMenu("Rendering"))
		{
			if (ImGui::MenuItem("Render Image"))
			{
				uiRenderingWindow = true;
			}
			ImGui::EndMenu();
		}
		ImGui::Separator();

		ImGui::SameLine(ImGui::GetWindowWidth() - 400);

		ImGui::PushItemWidth(100);
		if (ImGui::Combo("combo", (int*) &viewType, viewTypeArray, IM_ARRAYSIZE(viewTypeArray)))
			camera->dirty = true;
		ImGui::PopItemWidth();
		ImGui::Separator();
		ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f * deltaTime, 1.0f / deltaTime);
		ImGui::EndMainMenuBar();
	}

	if (uiRenderingWindow)
	{
		ImGui::SetNextWindowPosCenter();
		ImGui::Begin("Render Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

		ImGui::InputInt("Max Samples", &currentOption.maxSamples);
		ImGui::Checkbox("Enable Dof", &currentOption.enableDof);

		if (ImGui::Button("Render Start"))
		{
			gpuErrorCheck(cudaGraphicsMapResources(1, &viewResource));
			gpuErrorCheck(cudaGraphicsSubResourceGetMappedArray(&viewArray, viewResource, 0, 0));

			cudaResourceDesc viewCudaArrayResourceDesc;
			{
				viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
				viewCudaArrayResourceDesc.res.array.array = viewArray;
			}

			cudaSurfaceObject_t viewCudaSurfaceObject;
			gpuErrorCheck(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));
			{
				currentOption.surf = viewCudaSurfaceObject;
				currentOption.frame = 1;
				currentOption.isAccumulate = false;
				RenderKernel(camera, spheres, tree, currentOption);
			}
			gpuErrorCheck(cudaDestroySurfaceObject(viewCudaSurfaceObject));
			gpuErrorCheck(cudaGraphicsUnmapResources(1, &viewResource));
			cudaStreamSynchronize(0);
			currentOption.frame = currentOption.maxSamples;
			viewType = ViewType::IMAGE;
			uiRenderingWindow = false;
		}
		ImGui::SameLine();
		if (ImGui::Button("Cancel"))
		{
			uiRenderingWindow = false;
		}
		ImGui::End();
	}

	if (uiFileSaveDialog)
	{
		if (ImGuiFileDialog::Instance()->FileDialog("save", ".png", ".", "result.png"))
		{
			if (ImGuiFileDialog::Instance()->IsOk)
			{

				GLubyte *pixels = new GLubyte[3 * width*height];
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
				FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);

				{
					const unsigned bytesperpixel = FreeImage_GetBPP(image) / 8;
					const unsigned height = FreeImage_GetHeight(image);
					const unsigned pitch = FreeImage_GetPitch(image);
					const unsigned lineSize = FreeImage_GetLine(image);

					BYTE* line = FreeImage_GetBits(image);
					for (unsigned y = 0; y < height; ++y, line += pitch)
					{
						for (BYTE* pixel = line; pixel < line + lineSize; pixel += bytesperpixel)
						{
							std::swap(pixel[0], pixel[2]);
						}
					}
				}

				FreeImage_Save(FIF_PNG, image, ImGuiFileDialog::Instance()->GetFilepathName().c_str(), 0);
				FreeImage_Unload(image);
				delete pixels;
			}
			uiFileSaveDialog = false;
		}
	}

	if (uiObjLoadDialog)
	{
		if (ImGuiFileDialog::Instance()->FileDialog("Load", ".obj", ".", ""))
		{
			if (ImGuiFileDialog::Instance()->IsOk)
			{
				LoadObj(ImGuiFileDialog::Instance()->GetFilepathName().c_str());
			}
			uiObjLoadDialog = false;
		}
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


	glfwMakeContextCurrent(GLFWManager::GetWindow());
	glfwSwapBuffers(GLFWManager::GetWindow());
}

void CudaRenderer::LoadObj(const char* fileName)
{
	meshes.emplace_back(vec3(0), fileName);

	vector<vec3> verts;
	vector<vec3> norms;
	vector<Material> materials;
	vector<ivec3> vertexIndices;
	vector<ivec3> normalIndices;
	vector<int> materialIndices;

	for (auto & mesh : meshes)
	{
		int numVerts = verts.size();
		int numNorms = norms.size();
		int numMaterials = materials.size();

		verts.reserve(verts.size() + mesh.verts.size());
		norms.reserve(norms.size() + mesh.norms.size());
		materials.reserve(materials.size() + mesh.materials.size());
		vertexIndices.reserve(vertexIndices.size() + mesh.vertexIndices.size());
		normalIndices.reserve(normalIndices.size() + mesh.normalIndices.size());

		verts.insert(verts.end(), mesh.verts.begin(), mesh.verts.end());
		norms.insert(norms.end(), mesh.norms.begin(), mesh.norms.end());
		materials.insert(materials.end(), mesh.materials.begin(), mesh.materials.end());

		transform(mesh.vertexIndices.begin(), mesh.vertexIndices.end(), back_inserter(vertexIndices), [&](const ivec3& t) { return t + ivec3(numVerts); });
		transform(mesh.normalIndices.begin(), mesh.normalIndices.end(), back_inserter(normalIndices), [&](const ivec3& t) { return t + ivec3(numNorms); });
		transform(mesh.materialIndices.begin(), mesh.materialIndices.end(), back_inserter(materialIndices), [&](const int& t) { return t + numMaterials; });

	}
	if (tree)
		delete tree;
	tree = new KDTree(verts, vertexIndices, norms, normalIndices, materials, materialIndices);
	camera->dirty = true;
}

void CudaRenderer::HandleKeyboard(int key, int scancode, int action, int mods)
{

}

void CudaRenderer::HandleMouseClick(int button, int action, int mods)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else
		glfwSetInputMode(GLFWManager::GetWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void CudaRenderer::HandleMouseMotion(double xPos, double yPos)
{
	if (GLFWManager::IsMouseDown(GLFW_MOUSE_BUTTON_MIDDLE))
	{
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
		vec2 input = vec2(camera->height / 2 - yPos, camera->width / 2 - xPos);
		camera->UpdateCamera(deltaTime, vec2(0), input);
		GLFWManager::SetCursorToPos(camera->width / 2, camera->height / 2);
	}
}

void CudaRenderer::HandleResize(int width, int height)
{
	camera->UpdateScreen(width, height);
	glViewport(0, 0, width, height);
	cudaGraphicsUnregisterResource(viewResource);
	glBindTexture(GL_TEXTURE_2D, viewGLTexture);
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaGraphicsGLRegisterImage(&viewResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
}

