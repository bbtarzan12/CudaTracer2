#ifndef H_MEASURETIME
#define H_MEASURETIME
#include <string>
#include <chrono>

namespace MeasureTime
{

	class Timer
	{
	public:
		Timer();
		void Start(std::string message);
		void End(std::string message);

	private:
		std::chrono::steady_clock::time_point startPoint;
		bool isStart;
	};

}

#endif
