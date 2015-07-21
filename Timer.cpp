#include "Timer.h"

namespace hb {

	///Constructs a new \c Timer; the timer automatically starts running.
	Timer::Timer() {
		reset();
	}

	///Resets the time to zero; the timer automatically continues running.
	void Timer::reset() {
	#if defined WINDOWS
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		_frequency = freq.QuadPart;
		QueryPerformanceCounter(&_startingTime);
	#else
		_start = _start = std::chrono::high_resolution_clock::now();
	#endif
	}

	///Takes the time and prints it (in seconds) on the console.
	void Timer::stop() {
		long double time = getTime();
		std::cout << time << "s" << std::endl;
	}

	///Takes the time and returns it (in seconds).
	long double Timer::getTime() {
	#if defined WINDOWS
		LARGE_INTEGER endingTime;
		QueryPerformanceCounter(&endingTime);
		long double ticks = endingTime.QuadPart - _startingTime.QuadPart;
		return ticks / _frequency;
	#else
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::duration<long double>>(end - _start).count();
	#endif

	}

}