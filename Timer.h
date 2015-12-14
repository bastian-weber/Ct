#ifndef HB_TIMER
#define HB_TIMER

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#define WINDOWS
#endif

#include <chrono>
#include <iostream>

#if defined WINDOWS
#include <Windows.h>
#endif

namespace hb {

	///A timer that delivers precise time measurements under Windows and Linux.
	class Timer {
	public:
		Timer();
		void reset();
		void stop();
		long double getTime();
	private:
	#if defined WINDOWS
		LARGE_INTEGER startingTime;
		long double frequency;
	#else
		std::chrono::high_resolution_clock::time_point _start;
	#endif

	};

}

#endif