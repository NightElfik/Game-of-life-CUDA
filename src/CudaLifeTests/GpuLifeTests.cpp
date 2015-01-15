#include "stdafx.h"
#include "CudaLife/GpuLife.h"
#include "LifeTests.h"

namespace mf {

	TEST(GpuLife, LookupTable) {
		GpuLife gpuLife;

		std::vector<ubyte> lookupTable;
		lookupTable.resize(1 << 18);
		ASSERT_TRUE(cudaMemcpy(&lookupTable[0], gpuLife.getLookupTable(), 1 << 18 * sizeof(ubyte),
			cudaMemcpyDeviceToHost) == cudaError::cudaSuccess);

		LifeTests::checkLookupTable(&lookupTable[0]);
	}


	void prepareGpuForAllCombinations(GpuLife& gpuLife, bool bitLife) {
		gpuLife.resize(4 * 1 << 9, 4);
		gpuLife.allocBuffers(bitLife);

		size_t dataSize = 4 * 4 * 1 << 9;
		if (bitLife) {
			dataSize /= 8;
		}
		std::vector<ubyte> data;
		data.resize(dataSize);
		LifeTests::generateAllCombinationsData(&data[0], bitLife);

		ASSERT_TRUE(cudaMemcpy(bitLife ? gpuLife.bpcLifeData() : gpuLife.lifeData(), &data[0],
			dataSize * sizeof(ubyte), cudaMemcpyHostToDevice) == cudaError::cudaSuccess);
	}

	void evaluateGpuForAllCombinations(GpuLife& gpuLife, bool bitLife) {
		size_t dataSize = 4 * 4 * 1 << 9;
		if (bitLife) {
			dataSize /= 8;
		}
		std::vector<ubyte> data;
		data.resize(dataSize);

		ASSERT_TRUE(cudaMemcpy(&data[0], bitLife ? gpuLife.bpcLifeData() : gpuLife.lifeData(),
			dataSize * sizeof(ubyte), cudaMemcpyDeviceToHost) == cudaError::cudaSuccess);

		LifeTests::checkAllCombinationsData(&data[0], bitLife);
	}

	TEST(GpuLife, Baisc256Threads) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, false);
		ASSERT_TRUE(gpuLife.iterate(1, false, 256, false, 0, false));
		evaluateGpuForAllCombinations(gpuLife, false);
	}


	TEST(GpuLife, BitLookup256Threads1bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, true, 256, true, 1, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitLookup256Threads2bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, true, 256, true, 2, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitLookup256Threads4bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, true, 256, true, 4, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitLookup128Threads8bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, true, 128, true, 8, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}


	TEST(GpuLife, BitCounting256Threads1bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 256, true, 1, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCounting256Threads2bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 256, true, 2, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCounting256Threads4bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 256, true, 4, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCounting128Threads8bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 128, true, 8, false));
		evaluateGpuForAllCombinations(gpuLife, true);
	}


	TEST(GpuLife, BitCountingBig256Threads1bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 256, true, 1, true));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCountingBig128Threads2bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 128, true, 2, true));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCountingBig64Threads4bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 64, true, 4, true));
		evaluateGpuForAllCombinations(gpuLife, true);
	}

	TEST(GpuLife, BitCountingBig32Threads8bpt) {
		GpuLife gpuLife;
		prepareGpuForAllCombinations(gpuLife, true);
		ASSERT_TRUE(gpuLife.iterate(1, false, 32, true, 8, true));
		evaluateGpuForAllCombinations(gpuLife, true);
	}


	/// Negative tests

	TEST(GpuLife, TooManyThreads) {
		// Life world width: 2^5
		// 8 consecutive bytes: 2^5 / 8 = 2^2
		// data size = 4 * data width = 2^4
		// data size / threads count = 2^4 / 2^5 - ouch!
		GpuLife gpuLife;
		gpuLife.resize(1 << 5, 4);
		gpuLife.allocBuffers(false);

		ASSERT_FALSE(gpuLife.iterate(1, false, 1 << 5, true, 8, false));
	}

	TEST(GpuLife, TooManyThreadsBitLife) {
		// Life world width: 2^8
		// Bit life world width: 2^8 / 8 = 2^5
		// 8 consecutive bytes: 2^5 / 8 = 2^2
		// data size = 4 * data width = 2^4
		// data size / threads count = 2^4 / 2^5 - ouch!
		GpuLife gpuLife;
		gpuLife.resize(1 << 8, 4);
		gpuLife.allocBuffers(true);

		ASSERT_FALSE(gpuLife.iterate(1, false, 1 << 5, true, 8, false));
	}

	TEST(GpuLife, TooManyConecutiveBlocksPerThread) {
		// Life world width: 2^5
		// 2^6 consecutive bytes: 2^5 / 2^6 - ouch!
		GpuLife gpuLife;
		gpuLife.resize(1 << 5, 4);
		gpuLife.allocBuffers(false);

		ASSERT_FALSE(gpuLife.iterate(1, false, 1, true, 1 << 6, false));
	}

	TEST(GpuLife, TooManyConecutiveBlocksPerThreadBit) {
		// Life world width: 2^8
		// Bit life world width: 2^8 / 8 = 2^5
		// 2^6 consecutive bytes: 2^5 / 2^6 - ouch!
		GpuLife gpuLife;
		gpuLife.resize(1 << 8, 4);
		gpuLife.allocBuffers(true);

		ASSERT_FALSE(gpuLife.iterate(1, false, 1, true, 1 << 6, false));
	}

	TEST(GpuLife, TooManyConecutiveBlocksPerThreadBitBigBlocks) {
		// Life world width: 2^8
		// Bit life world width: 2^8 / 8 = 2^5
		// 2^4 consecutive blocks per 4 bytes: 2^5 / (2^3 * 4) - ouch!
		GpuLife gpuLife;
		gpuLife.resize(1 << 8, 4);
		gpuLife.allocBuffers(true);

		ASSERT_FALSE(gpuLife.iterate(1, false, 1, true, 1 << 4, true));
	}


}