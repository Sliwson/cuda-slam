#include "loader.h"
#include <filesystem>

namespace Common
{
	AssimpCloudLoader::AssimpCloudLoader(std::string const& path)
	{
		LoadModel(path);

#ifdef _DEBUG
		printf("Loaded %zd clouds from file %s\n", clouds.size(), path.c_str());
		if (clouds.size() > 0)
		{
			printf("Sizes:\n");
			for (int i = 0; i < clouds.size(); i++)
				printf("%d: %zd\n", i, clouds[i].size());
		}
#endif

	}

	void AssimpCloudLoader::LoadModel(std::string const& path)
	{
		Assimp::Importer importer;
		const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);

		if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		{
#ifdef _DEBUG
			printf("%s\n", importer.GetErrorString());
#endif
			return;
		}

		ProcessNode(scene->mRootNode, scene);
	}

	void AssimpCloudLoader::ProcessNode(aiNode* node, const aiScene* scene)
	{
		for (unsigned int i = 0; i < node->mNumMeshes; i++)
		{
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			clouds.push_back(ProcessMesh(mesh, scene));
		}

		for (unsigned int i = 0; i < node->mNumChildren; i++)
			ProcessNode(node->mChildren[i], scene);
	}

	std::vector<Point_f> AssimpCloudLoader::ProcessMesh(aiMesh* mesh, const aiScene* scene)
	{
		std::vector<Point_f> cloud(mesh->mNumVertices);

		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
			cloud[i] = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };

		return cloud;
	}
}

