#include "Mesh.h"

#include "MeasureTime.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#pragma region Mesh

std::istream& safeGetline(std::istream& is, std::string& t)
{
	t.clear();

	// The characters in the stream are read one-by-one using a std::streambuf.
	// That is faster than reading them one-by-one using the std::istream.
	// Code that uses streambuf this way must be guarded by a sentry object.
	// The sentry object performs various tasks,
	// such as thread synchronization and updating the stream state.

	std::istream::sentry se(is, true);
	std::streambuf* sb = is.rdbuf();

	for (;;)
	{
		int c = sb->sbumpc();
		switch (c)
		{
			case '\n':
				return is;
			case '\r':
				if (sb->sgetc() == '\n')
					sb->sbumpc();
				return is;
			case EOF:
				// Also handle the case when the last line has no line ending
				if (t.empty())
					is.setstate(std::ios::eofbit);
				return is;
			default:
				t += (char) c;
		}
	}
}

std::vector<std::string> tokenizeString(std::string str)
{
	std::stringstream strstr(str);
	std::istream_iterator<std::string> it(strstr);
	std::istream_iterator<std::string> end;
	std::vector<std::string> results(it, end);
	return results;
}

Mesh::Mesh(vec3 position /*= vec3(0)*/, string fileName /*= ""*/, int materialID /*= 0*/)
{
	auto timer = MeasureTime::Timer();
	timer.Start("[Mesh] Load Start");
	this->position = position;

	ifstream ifile;
	string line;
	ifile.open(fileName.c_str());

	while (safeGetline(ifile, line))
	{
		vector<string> tokens = tokenizeString(line);

		if (!tokens.empty() && strcmp(tokens[0].c_str(), "v") == 0)
		{
			verts.emplace_back(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (!tokens.empty() && strcmp(tokens[0].c_str(), "f") == 0)
		{
			char* findex1 = strtok(const_cast<char*>(tokens[1].c_str()), "/");
			char* findex2 = strtok(const_cast<char*>(tokens[2].c_str()), "/");
			char* findex3 = strtok(const_cast<char*>(tokens[3].c_str()), "/");
			tris.emplace_back(atoi(findex1) - 1, atoi(findex2) - 1, atoi(findex3) - 1);
		}
	}
	timer.End("[Mesh] Load Success");
}

#pragma endregion Mesh