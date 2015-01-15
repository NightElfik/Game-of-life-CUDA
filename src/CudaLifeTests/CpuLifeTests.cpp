#include "stdafx.h"
#include "CudaLife/CpuLife.h"
#include "LifeTests.h"

namespace mf {


	TEST(CpuLife, LookupTable) {
		CpuLife cpuLife;
		LifeTests::checkLookupTable(cpuLife.getLookupTable());
	}


	void prepareCpuForAllCombinations(CpuLife& cpuLife, bool bitLife) {
		cpuLife.resize(4 * 1 << 9, 4);
		cpuLife.allocBuffers(bitLife);
		LifeTests::generateAllCombinationsData(bitLife ? cpuLife.bpcLifeData() : cpuLife.lifeData(), bitLife);
	}

	void evaluateCpuForAllCombinations(CpuLife& cpuLife, bool bitLife) {
		LifeTests::checkAllCombinationsData(bitLife ? cpuLife.bpcLifeData() : cpuLife.lifeData(), bitLife);
	}

	TEST(CpuLife, Serial) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, false);
		ASSERT_TRUE(cpuLife.iterateSerial(1));
		evaluateCpuForAllCombinations(cpuLife, false);
	}

	TEST(CpuLife, ParallelStaticFunc) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, false);
		ASSERT_TRUE(cpuLife.iterateParallelStaticFunc(1));
		evaluateCpuForAllCombinations(cpuLife, false);
	}

	TEST(CpuLife, ParallelLambdaFunc) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, false);
		ASSERT_TRUE(cpuLife.iterateParallelLambda(1));
		evaluateCpuForAllCombinations(cpuLife, false);
	}


	TEST(CpuLife, BitLookup1bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 1, true, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitLookup2bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 2, true, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitLookup4bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 4, true, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitLookup8bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 8, true, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}


	TEST(CpuLife, BitCounting1bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 1, false, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCounting2bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 2, false, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCounting4bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 4, false, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCounting8bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 8, false, false));
		evaluateCpuForAllCombinations(cpuLife, true);
	}


	TEST(CpuLife, BitCountingBig1bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 1, false, true));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCountingBig2bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 2, false, true));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCountingBig4bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 4, false, true));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

	TEST(CpuLife, BitCountingBig8bpt) {
		CpuLife cpuLife;
		prepareCpuForAllCombinations(cpuLife, true);
		ASSERT_TRUE(cpuLife.iterateBitLife(1, 8, false, true));
		evaluateCpuForAllCombinations(cpuLife, true);
	}

}