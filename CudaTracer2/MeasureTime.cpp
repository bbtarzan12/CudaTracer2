#include "MeasureTime.h"

#include <iostream>

using namespace std;

MeasureTime::Timer::Timer()
{
	isStart = false;
}

void MeasureTime::Timer::Start(std::string message)
{
	cout << message << endl;
	startPoint = std::chrono::high_resolution_clock::now();
	isStart = true;
}

void MeasureTime::Timer::End(std::string message)
{
	if (!isStart)
	{
		cout << "[Timer] Please Start the Timer" << endl;
		return;
	}

	auto endPoint = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = endPoint - startPoint;
	cout << message << " Elapsed time: " << elapsed.count() << endl;
}
