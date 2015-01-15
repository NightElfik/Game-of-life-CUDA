#pragma once
#include "CpuLife.h"
#include "GpuLife.h"

namespace mf {

	struct BenchmarkSettings {
		size_t lifeIters;
		size_t maxLifeWorldSize;
		size_t maxLifeWorldCpuSerialSize;
		size_t gpuMeasurementIterations;
		size_t cpuMeasurementIterations;
		uint minBytesPerThread;
		uint maxBytesPerThread;
		ushort minTreadsCount;
		ushort maxTreadsCount;
		size_t initialWorldWidth;
		size_t initialWorldHeight;

		size_t individualIterationsCount;
		size_t individualItesCpuCb;
		size_t individualItesGpuCb;
		size_t individualItesTpb;
	};

	template<typename NoCppFileNeeded = int>
	class TBenchmark {

		BenchmarkSettings m_benchSettings;

	public:

		TBenchmark() {
			setSettingsToDefaultValues();
		}

		const BenchmarkSettings& getSettings() const {
			return m_benchSettings;
		}

		BenchmarkSettings& settings() {
			return m_benchSettings;
		}

		void setSettingsToDefaultValues() {
			m_benchSettings.lifeIters = 16;
			m_benchSettings.maxLifeWorldSize = 1ull << 32;
			m_benchSettings.maxLifeWorldCpuSerialSize = 1ull << 32;
			m_benchSettings.gpuMeasurementIterations = 11;
			m_benchSettings.cpuMeasurementIterations = 5;
			m_benchSettings.minBytesPerThread = 1;
			m_benchSettings.maxBytesPerThread = 128;
			m_benchSettings.minTreadsCount = 32;
			m_benchSettings.maxTreadsCount = 1024;
			m_benchSettings.initialWorldWidth = 1024;
			m_benchSettings.initialWorldHeight = 8;

			m_benchSettings.individualIterationsCount = 64;
			m_benchSettings.individualItesCpuCb = 256;
			m_benchSettings.individualItesGpuCb = 8;
			m_benchSettings.individualItesTpb = 128;
		}

		void setSettingsToDebugValues() {
			//m_benchSettings.lifeIters = 1;
			m_benchSettings.maxLifeWorldSize = 1ull << 28;
			m_benchSettings.maxLifeWorldCpuSerialSize = 1ull << 26;
			//m_benchSettings.gpuMeasurementIterations = 1;
			//m_benchSettings.cpuMeasurementIterations = 1;
			//m_benchSettings.minBytesPerThread = 128;
			//m_benchSettings.maxBytesPerThread = 128;
			//m_benchSettings.minTreadsCount = 128;
			//m_benchSettings.maxTreadsCount = 128;
			//m_benchSettings.initialWorldWidth = 1ull << 16;
			//m_benchSettings.initialWorldHeight = 1ull << 16;
		}

		/// Runs the main benchmark method that runs any requested sub-benchmarks.
		void runBenchmark(bool verbose, bool runCpuBench, bool runGpuBench, bool runIndividualInstancesFirst,
				const std::string& outFilePath) {

			if (verbose) {
				std::cout << std::endl << "Benchmark started" << std::endl;
			}


			std::ofstream resultsStream;
			resultsStream.open(outFilePath);
			assert(resultsStream.good());

			if (runIndividualInstancesFirst) {
				runCpuIndividualIterationsTimingBenchmark(verbose, resultsStream);
			}

			if (verbose) {
				std::cout << std::endl << "Iterations count: " << m_benchSettings.lifeIters << std::endl;
			}

			resultsStream << "Times per iteration [ms]" << std::endl
				<< "CPU median of:;" << m_benchSettings.cpuMeasurementIterations << ";runs" << std::endl
				<< "GPU median of:;" << m_benchSettings.gpuMeasurementIterations << ";runs" << std::endl
				<< "Life iterations count:;" << m_benchSettings.lifeIters << ";(avg per iteration)" << std::endl;
			resultsStream << std::endl;


			if (runCpuBench) {
				runCpuBenchmark(verbose, resultsStream);
			}

			if (runGpuBench) {
				runGpuBenchmark(verbose, resultsStream);
			}

			if (verbose) {
				std::cout << "Benchmark done..." << std::endl;
			}
		}

		/// Runs individual test for measuring time per iteration for selected implementations.
		void runCpuIndividualIterationsTimingBenchmark(bool verbose, std::ostream& resultsStream) {
			{
				CpuLife cpuLife;

				runCpuIndividualIterationsTimingBenchmarkInstance(verbose, "CPU bit counting", resultsStream,
					m_benchSettings.cpuMeasurementIterations,
					[&cpuLife] (size_t w, size_t h){ cpuLife.resize(w, h); },
					[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(true); },
					[&cpuLife, this] {
						return cpuLife.iterateBitLife(1, m_benchSettings.individualItesCpuCb, false, false);
					});

				runCpuIndividualIterationsTimingBenchmarkInstance(verbose, "CPU lookup", resultsStream,
					m_benchSettings.cpuMeasurementIterations,
					[&cpuLife] (size_t w, size_t h){ cpuLife.resize(w, h); },
					[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(true); },
					[&cpuLife, this] {
						return cpuLife.iterateBitLife(1, m_benchSettings.individualItesCpuCb, false, false);
					});
			}

			{
				GpuLife gpuLife;

				runCpuIndividualIterationsTimingBenchmarkInstance(verbose, "GPU bit counting",
					resultsStream, m_benchSettings.gpuMeasurementIterations,
					[&gpuLife] (size_t w, size_t h){ gpuLife.resize(w, h); },
					[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(true); },
					[&gpuLife, this] {
						return gpuLife.iterate(m_benchSettings.lifeIters, false, ushort(m_benchSettings.individualItesTpb),
							true, uint(m_benchSettings.individualItesGpuCb), false);
					});

				runCpuIndividualIterationsTimingBenchmarkInstance(verbose, "GPU lookup",
					resultsStream, m_benchSettings.gpuMeasurementIterations,
					[&gpuLife] (size_t w, size_t h){ gpuLife.resize(w, h); },
					[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(true); },
					[&gpuLife, this] {
						return gpuLife.iterate(m_benchSettings.lifeIters, true, ushort(m_benchSettings.individualItesTpb),
							true, uint(m_benchSettings.individualItesGpuCb), false);
					});
			}
		}

		/// Runs standard benchmark of CPU life algorithms.
		void runCpuBenchmark(bool verbose, std::ostream& resultsStream) {
			CpuLife cpuLife;

			resultsStream << std::endl << "CPU" << std::endl << std::endl;
			resultsStream << ";;;Consecutive blocks:";

			uint minBpt = m_benchSettings.minBytesPerThread;
			uint maxBpt = m_benchSettings.maxBytesPerThread;

			for (size_t bpt = minBpt; bpt <= maxBpt; bpt *= 2) {
				resultsStream << ";" << bpt << ";" << bpt << ";" << bpt;
			}
			resultsStream << std::endl;

			resultsStream << "World size (wid x hei);Serial;Parallel static func;Parallel lambda";

			for (size_t bpt = minBpt; bpt <= maxBpt; bpt *= 2) {
				resultsStream << ";Bit (counting)";
				resultsStream << ";Bit (counting big)";
				resultsStream << ";Bit (lookup)";
			}
			resultsStream << std::endl;

			std::vector<float> cpuTimes;
			cpuTimes.resize(m_benchSettings.cpuMeasurementIterations);

			for (size_t worldWidth = m_benchSettings.initialWorldWidth,
						worldHeight = m_benchSettings.initialWorldHeight;;) {

				if (worldWidth * worldHeight > m_benchSettings.maxLifeWorldSize) {
					break;
				}

				resultsStream << (worldWidth * worldHeight) << ";";

				if (verbose) {
					std::cout << std::endl << "Current world: " << worldWidth << " x " << worldHeight << " = "
						<< (worldWidth * worldHeight) << std::endl;
				}

				cpuLife.resize(worldWidth, worldHeight);

				if (worldWidth * worldHeight <= m_benchSettings.maxLifeWorldCpuSerialSize) {
					runMeasurementInstance(verbose, "CPU serial",
						resultsStream, m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
						[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(false); },
						[&cpuLife, this] { return cpuLife.iterateSerial(m_benchSettings.lifeIters); });
				}
				else {
					resultsStream << ";";
				}

				runMeasurementInstance(verbose, "CPU parallel static func",
					resultsStream,m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
					[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(false); },
					[&cpuLife, this] { return cpuLife.iterateParallelStaticFunc(m_benchSettings.lifeIters); });

				runMeasurementInstance(verbose, "CPU parallel lambda",
					resultsStream, m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
					[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(false); },
					[&cpuLife, this] { return cpuLife.iterateParallelLambda(m_benchSettings.lifeIters); });

				for (size_t bpt = minBpt; bpt <= maxBpt; bpt *= 2) {
					runMeasurementInstance(verbose, "CPU bit life bit counting cpt: " + std::to_string(bpt),
						resultsStream, m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
						[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(true); },
						[&cpuLife, bpt, this] {
							return cpuLife.iterateBitLife(m_benchSettings.lifeIters, bpt, false, false);
						});

					runMeasurementInstance(verbose, "CPU bit life bit counting big chunks cpt: " + std::to_string(bpt),
						resultsStream, m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
						[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(true); },
						[&cpuLife, bpt, this] {
							return cpuLife.iterateBitLife(m_benchSettings.lifeIters, bpt, false, true);
						});

					runMeasurementInstance(verbose, "CPU bit life lookup: " + std::to_string(bpt),
						resultsStream, m_benchSettings.cpuMeasurementIterations, float(m_benchSettings.lifeIters),
						[&cpuLife] { cpuLife.freeBuffers(); return cpuLife.allocBuffers(true); },
						[&cpuLife, bpt, this] {
							return cpuLife.iterateBitLife(m_benchSettings.lifeIters, bpt, true, false);
						});
				}

				if (worldWidth <= worldHeight) {
					worldWidth *= 2;
				}
				else {
					worldHeight *= 2;
				}

				resultsStream << std::endl;
			}

			resultsStream << std::endl;
		}

		/// Runs standard benchmark of GPU life algorithms.
		void runGpuBenchmark(bool verbose, std::ostream& resultsStream) {
			GpuLife gpuLife;

			resultsStream << std::endl << "GPU" << std::endl << std::endl;
			std::vector<float> gpuTimes;
			gpuTimes.resize(m_benchSettings.gpuMeasurementIterations);

			uint minBpt = m_benchSettings.minBytesPerThread;
			uint maxBpt = m_benchSettings.maxBytesPerThread;

			for (ushort threadsCount = m_benchSettings.minTreadsCount;
					threadsCount <= m_benchSettings.maxTreadsCount;
					threadsCount *= 2) {

				if (verbose) {
					std::cout << std::endl << "Threads count: " << threadsCount << std::endl;
				}

				resultsStream << "Threads count:;" << std::endl << threadsCount<< ";;;Consecutive blocks:";

				for (uint bpt = minBpt; bpt <= maxBpt; bpt *= 2) {
					resultsStream << ";" << bpt << ";" << bpt << ";" << bpt;
				}
				resultsStream << std::endl;

				resultsStream << "World size (wid x hei);;;Byte per cell";

				for (uint bpt = minBpt; bpt <= maxBpt; bpt *= 2) {
					resultsStream << ";Bit (counting)";
					resultsStream << ";Bit (counting big)";
					resultsStream << ";Bit (lookup)";
				}
				resultsStream << std::endl;

				for (size_t worldWidth = m_benchSettings.initialWorldWidth,
						worldHeight = m_benchSettings.initialWorldHeight;;) {

					if (worldWidth * worldHeight > m_benchSettings.maxLifeWorldSize) {
						break;
					}

					resultsStream << (worldWidth * worldHeight) << ";;;";

					if (verbose) {
						std::cout << std::endl << "Current world: " << worldWidth << " x " << worldHeight << " = "
							<< (worldWidth * worldHeight) << std::endl;
					}

					gpuLife.resize(worldWidth, worldHeight);

					runMeasurementInstance(verbose, "GPU simple",
						resultsStream, m_benchSettings.gpuMeasurementIterations, float(m_benchSettings.lifeIters),
						[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(false); },
						[&gpuLife, threadsCount, this] {
							return gpuLife.iterate(m_benchSettings.lifeIters, false, threadsCount, false, 0, false);
						});

					for (uint bpt = minBpt; bpt <= maxBpt; bpt *= 2) {

						runMeasurementInstance(verbose, "GPU bit life bit counting bpt: " + std::to_string(bpt),
							resultsStream, m_benchSettings.gpuMeasurementIterations, float(m_benchSettings.lifeIters),
							[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(true); },
							[&gpuLife, threadsCount, bpt, this] {
								return gpuLife.iterate(m_benchSettings.lifeIters, false, threadsCount, true, bpt,
									false);
							});

						runMeasurementInstance(verbose, "GPU bit life bit counting big chunks bpt: " + std::to_string(bpt),
							resultsStream, m_benchSettings.gpuMeasurementIterations, float(m_benchSettings.lifeIters),
							[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(true); },
							[&gpuLife, threadsCount, bpt, this] {
								return gpuLife.iterate(m_benchSettings.lifeIters, false, threadsCount, true, bpt, true);
							});

						runMeasurementInstance(verbose, "GPU bit life lookup bpt: " + std::to_string(bpt),
							resultsStream, m_benchSettings.gpuMeasurementIterations, float(m_benchSettings.lifeIters),
							[&gpuLife] { gpuLife.freeBuffers(); return gpuLife.allocBuffers(true); },
							[&gpuLife, threadsCount, bpt, this] {
								return gpuLife.iterate(m_benchSettings.lifeIters, true, threadsCount, true, bpt, false);
							});
					}

					if (worldWidth <= worldHeight) {
						worldWidth *= 2;
					}
					else {
						worldHeight *= 2;
					}

					resultsStream << std::endl;
				}

				resultsStream << std::endl;
			}
		}


	private:

		/// Measures given measure function and reports median time from given number of separated iterations.
		/// Reset function is run before each measured function but it is not measured by itself.
		void runMeasurementInstance(bool verbose, const std::string& name, std::ostream& resultsStream, size_t iterations,
				float timesFactor, std::function<bool()> resetFunc, std::function<bool()> measuredFunc) {

			if (verbose) {
				std::cout << name;
			}

			std::vector<float> times;
			times.resize(iterations);

			for (size_t i = 0; i < iterations; ++i) {
				float time;
				if (resetFunc()) {
					auto t1 = std::chrono::high_resolution_clock::now();
					bool result = measuredFunc();
					auto t2 = std::chrono::high_resolution_clock::now();
					time = result
						? (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f)
						: std::numeric_limits<float>::quiet_NaN();
				}
				else {
					time = std::numeric_limits<float>::quiet_NaN();
				}

				times[i] = time / timesFactor;
			}

			std::sort(times.begin(), times.end());
			volatile float median = times[iterations / 2];
			if (median == median) {  // Write result if median is not NaN.
				resultsStream << median;
			}
			resultsStream << ";";

			size_t length = 60;
			if (verbose) {
				std::cout << std::string(std::max(2ull, length - name.length()), ' ') << median << " ms" << std::endl;
			}
		}

		void runMeasurementIndividualIterationsInstance(bool verbose,
				std::ostream& resultsStream, size_t iterations,
				std::function<bool()> resetFunc, std::function<bool()> oneIterFunc) {

			size_t iterMax = m_benchSettings.individualIterationsCount;

			std::vector<std::vector<float>> times;
			times.resize(iterMax);
			for (std::vector<float>& v : times) {
				v.resize(iterations);
			}

			for (size_t i = 0; i < iterations; ++i) {
				if (resetFunc()) {
					for (size_t iter = 0; iter < iterMax; ++iter) {
						auto t1 = std::chrono::high_resolution_clock::now();
						bool result = oneIterFunc();
						auto t2 = std::chrono::high_resolution_clock::now();
						times[iter][i] = result
							? (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0f)
							: std::numeric_limits<float>::quiet_NaN();
					}
				}
				else {
					for (size_t iter = 0; iter < iterMax; ++iter) {
						times[iter][i] = std::numeric_limits<float>::quiet_NaN();
					}
				}
			}

			for (std::vector<float>& v : times) {
				std::sort(v.begin(), v.end());
				volatile float median = v[iterations / 2];
				if (median == median) {  // Write result if median is not NaN.
					resultsStream << median;
				}
				resultsStream << ";";
			}

			resultsStream << std::endl;
		}


		void runCpuIndividualIterationsTimingBenchmarkInstance(bool verbose, const std::string& name,
				std::ostream& resultsStream, size_t iterations, std::function<void(size_t, size_t)> resizeFunc,
				std::function<bool()> resetFunc, std::function<bool()> oneIterFunc) {

			resultsStream << std::endl << name << std::endl;
			resultsStream << "Iteration number:";

			resultsStream << "World size (wid x hei);" << std::endl;

			for (size_t i = 1; i <= m_benchSettings.individualIterationsCount; ++i) {
				resultsStream << ";" << i;
			}
			resultsStream << std::endl;

			if (verbose) {
				std::cout << name << std::endl;
			}

			for (size_t worldWidth = m_benchSettings.initialWorldWidth,
						worldHeight = m_benchSettings.initialWorldHeight;;) {

				if (worldWidth * worldHeight > m_benchSettings.maxLifeWorldSize) {
					break;
				}

				resultsStream << (worldWidth * worldHeight) << ";";

				if (verbose) {
					std::cout << "Current world: " << worldWidth << " x " << worldHeight << " = "
						<< (worldWidth * worldHeight) << std::endl;
				}

				resizeFunc(worldWidth, worldHeight);

				runMeasurementIndividualIterationsInstance(verbose, resultsStream,
					m_benchSettings.cpuMeasurementIterations, resetFunc, oneIterFunc);

				if (worldWidth <= worldHeight) {
					worldWidth *= 2;
				}
				else {
					worldHeight *= 2;
				}
			}

			resultsStream << std::endl;
		}

	};

	typedef TBenchmark<> Benchmark;
}