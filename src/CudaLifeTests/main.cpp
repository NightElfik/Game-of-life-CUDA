#include "stdafx.h"

int main(int argc, char* argv[]) {

	//testing::GTEST_FLAG(filter) = "CpuLife.*";

	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

