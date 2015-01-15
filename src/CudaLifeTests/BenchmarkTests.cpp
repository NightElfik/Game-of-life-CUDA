#include "stdafx.h"
#include "CudaLife/Benchmark.h"

namespace mf {

	// Those tests are more like safety guard to see if invocation of benchmarks does not throw any exceptions etc.

	TEST(Benchmark, Cpu) {
		Benchmark bench;
		BenchmarkSettings& settings = bench.settings();
		settings.lifeIters = 1;
		settings.cpuMeasurementIterations = 1;
		settings.initialWorldWidth = 1 << 14;
		settings.initialWorldHeight = 1 << 2;
		settings.maxLifeWorldSize = 1 << 16;
		settings.maxLifeWorldCpuSerialSize = 1 << 16;
		settings.minBytesPerThread = 4;
		settings.maxBytesPerThread = 4;

		std::ostringstream resultsStream;
		bench.runCpuBenchmark(false, resultsStream);
	}

	TEST(Benchmark, Gpu) {
		Benchmark bench;
		BenchmarkSettings& settings = bench.settings();
		settings.lifeIters = 1;
		settings.gpuMeasurementIterations = 1;
		settings.initialWorldWidth = 1 << 14;
		settings.initialWorldHeight = 1 << 2;
		settings.maxLifeWorldSize = 1 << 16;
		settings.maxLifeWorldCpuSerialSize = 1 << 16;
		settings.minTreadsCount = 64;
		settings.maxTreadsCount = 64;
		settings.minBytesPerThread = 4;
		settings.maxBytesPerThread = 4;

		std::ostringstream resultsStream;
		bench.runCpuBenchmark(false, resultsStream);
	}

}