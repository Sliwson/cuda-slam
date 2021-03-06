#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "_common.h"

namespace Common
{
	class AssimpCloudLoader
	{
	public:
		AssimpCloudLoader(std::string const& path);
		std::vector<Point_f> GetMergedCloud() const;
		std::vector<Point_f> GetCloud(int idx) const { return clouds[idx]; }
		int GetCloudCount() const { return static_cast<int>(clouds.size()); }

	private:

		void LoadModel(std::string const& path);
		void ProcessNode(aiNode* node, const aiScene* scene);
		std::vector<Point_f> ProcessMesh(aiMesh* mesh, const aiScene* scene);

		std::vector<std::vector<Point_f>> clouds;
	};
}
