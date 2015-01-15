#include "CudaLifeFunctions.h"

#include <assert.h>
#include <cuda_runtime.h>

#include "OpenGlCudaHelper.h"

// CUDA facts:
//
// On devices of compute capability 2.x and beyond, 32-bit integer multiplication is natively supported,
// but 24-bit integer multiplication is not. __[u]mul24 is therefore implemented using multiple instructions
// and should not be used.
//
// Integer division and modulo operation are costly: below 20 instructions on devices of compute capability 2.x and
// higher. They can be replaced with bitwise operations in some cases: If n is a power of 2, (i/n) is equivalent to
// (i>>log2(n)) and (i%n) is equivalent to (i&(n-1)); the compiler will perform these conversions if n is literal.

namespace mf {
	/// CUDA kernel for simple byte-per-cell world evaluation.
	///
	/// @param lifeData  Linearized 2D array of life data with byte-per-cell density.
	/// @param worldWidth  Width of life world in cells (bytes).
	/// @param worldHeight  Height of life world in cells (bytes).
	/// @param resultLifeData  Result buffer in the same format as input.
	__global__ void simpleLifeKernel(const ubyte* lifeData, uint worldWidth, uint worldHeight, ubyte* resultLifeData) {
		uint worldSize = worldWidth * worldHeight;

		for (uint cellId = blockIdx.x * blockDim.x + threadIdx.x;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x) {

			uint x = cellId % worldWidth;
			uint yAbs = cellId - x;

			uint xLeft = (x + worldWidth - 1) % worldWidth;
			uint xRight = (x + 1) % worldWidth;

			uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
			uint yAbsDown = (yAbs + worldWidth) % worldSize;

			// Count alive cells.
			uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp]
				+ lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
				+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

			resultLifeData[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
		}
	}

	/// Runs a kernel for simple byte-per-cell world evaluation.
	bool runSimpleLifeKernel(ubyte*& d_lifeData, ubyte*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight,
			size_t iterationsCount, ushort threadsCount) {

		if ((worldWidth * worldHeight) % threadsCount != 0) {
			return false;
		}

		size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
		ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

		for (size_t i = 0; i < iterationsCount; ++i) {
			simpleLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight),
				d_lifeDataBuffer);
			std::swap(d_lifeData, d_lifeDataBuffer);
		}
		checkCudaErrors(cudaDeviceSynchronize());

		return true;
	}

	/// CUDA kernel for rendering of life world on the screen.
	/// This kernel transforms bit-per-cell life world to ARGB screen buffer.
	__global__ void displayLifeKernel(const ubyte* lifeData, uint worldWidth, uint worldHeight, uchar4* destination,
			int destWidth, int detHeight, int2 displacement, double zoomFactor, int multisample, bool simulateColors,
			bool cyclic, bool bitLife) {

		uint pixelId = blockIdx.x * blockDim.x + threadIdx.x;

		int x = (int)floor(((int)(pixelId % destWidth) - displacement.x) * zoomFactor);
		int y = (int)floor(((int)(pixelId / destWidth) - displacement.y) * zoomFactor);

		if (cyclic) {
			x = ((x % (int)worldWidth) + worldWidth) % worldWidth;
			y = ((y % (int)worldHeight) + worldHeight) % worldHeight;
		}
		else if (x < 0 || y < 0 || x >= worldWidth || y >= worldHeight) {
			destination[pixelId].x = 127;
			destination[pixelId].y = 127;
			destination[pixelId].z = 127;
			return;
		}

		int value = 0;  // Start at value - 1.
		int increment = 255 / (multisample * multisample);

		if (bitLife) {
			for (int dy = 0; dy < multisample; ++dy) {
				int yAbs = (y + dy) * worldWidth;
				for (int dx = 0; dx < multisample; ++dx) {
					int xBucket = yAbs + x + dx;
					value += ((lifeData[xBucket >> 3] >> (7 - (xBucket & 0x7))) & 0x1) * increment;
				}
			}
		}
		else {
			for (int dy = 0; dy < multisample; ++dy) {
				int yAbs = (y + dy) * worldWidth;
				for (int dx = 0; dx < multisample; ++dx) {
					value += lifeData[yAbs + (x + dx)] * increment;
				}
			}
		}

		bool isNotOnBoundary = !cyclic || !(x == 0 || y == 0);

		if (simulateColors) {
			if (value > 0) {
				if (destination[pixelId].w > 0) {
					// Stayed alive - get darker.
					if (destination[pixelId].y > 63) {
						if (isNotOnBoundary) {
							--destination[pixelId].x;
						}
						--destination[pixelId].y;
						--destination[pixelId].z;
					}
				}
				else {
					// Born - full white color.
					destination[pixelId].x = 255;
					destination[pixelId].y = 255;
					destination[pixelId].z = 255;
				}
			}
			else {
				if (destination[pixelId].w > 0) {
					// Died - dark green.
					if (isNotOnBoundary) {
						destination[pixelId].x = 0;
					}
					destination[pixelId].y = 128;
					destination[pixelId].z = 0;
				}
				else {
					// Stayed dead - get darker.
					if (destination[pixelId].y > 8) {
						if (isNotOnBoundary) {
						}
						destination[pixelId].y -= 8;
					}
				}
			}
		}
		else {
			destination[pixelId].x = isNotOnBoundary ? value : 255;
			destination[pixelId].y = value;
			destination[pixelId].z = value;
		}

		// Save last state of the cell to the alpha channel that is not used in rendering.
		destination[pixelId].w = value;
	}

	/// Runs a kernel for rendering of life world on the screen.
	void runDisplayLifeKernel(const ubyte* d_lifeData, size_t worldWidth, size_t worldHeight, uchar4* destination,
			int destWidth, int destHeight, int displacementX, int displacementY, int zoom, bool simulateColors,
			bool cyclic, bool bitLife) {

		ushort threadsCount = 256;
		assert((worldWidth * worldHeight) % threadsCount == 0);
		size_t reqBlocksCount = (destWidth * destHeight) / threadsCount;
		assert(reqBlocksCount < 65536);
		ushort blocksCount = (ushort)reqBlocksCount;

		int multisample = std::min(4, (int)std::pow(2, std::max(0, zoom)));
		displayLifeKernel<<<blocksCount, threadsCount>>>(d_lifeData, uint(worldWidth), uint(worldHeight), destination,
			destWidth, destHeight, make_int2(displacementX, displacementY), std::pow(2, zoom),
			multisample, zoom > 1 ? false : simulateColors, cyclic, bitLife);
		checkCudaErrors(cudaDeviceSynchronize());
	}


	/// CUDA kernel that encodes byte-per-cell data to bit-per-cell data.
	/// Needs to be invoked for each byte in encoded data (cells / 8).
	__global__ void bitLifeEncodeKernel(const ubyte* lifeData, size_t encWorldSize, ubyte* resultEncodedLifeData) {

		for (size_t outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
				outputBucketId < encWorldSize;
				outputBucketId += blockDim.x * gridDim.x) {

			size_t cellId = outputBucketId << 3;

			ubyte result = lifeData[cellId] << 7 | lifeData[cellId + 1] << 6 | lifeData[cellId + 2] << 5
				| lifeData[cellId + 3] << 4 | lifeData[cellId + 4] << 3 | lifeData[cellId + 5] << 2
				| lifeData[cellId + 6] << 1 | lifeData[cellId + 7];

			resultEncodedLifeData[outputBucketId] = result;
		}

	}

	/// Runs a kernel that encodes byte-per-cell data to bit-per-cell data.
	void runBitLifeEncodeKernel(const ubyte* d_lifeData, uint worldWidth, uint worldHeight, ubyte* d_encodedLife) {

		assert(worldWidth % 8 == 0);
		size_t worldEncDataWidth = worldWidth / 8;
		size_t encWorldSize = worldEncDataWidth * worldHeight;

		ushort threadsCount = 256;
		assert(encWorldSize % threadsCount == 0);
		size_t reqBlocksCount = encWorldSize / threadsCount;
		ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

		bitLifeEncodeKernel<<<blocksCount, threadsCount>>>(d_lifeData, encWorldSize, d_encodedLife);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/// CUDA kernel that decodes data from bit-per-cell to byte-per-cell format.
	/// Needs to be invoked for each byte in encoded data (cells / 8).
	__global__ void bitLifeDecodeKernel(const ubyte* encodedLifeData, uint encWorldSize, ubyte* resultDecodedlifeData) {

		for (uint outputBucketId = blockIdx.x * blockDim.x + threadIdx.x;
				outputBucketId < encWorldSize;
				outputBucketId += blockDim.x * gridDim.x) {

			uint cellId = outputBucketId << 3;
			ubyte dataBucket = encodedLifeData[outputBucketId];

			resultDecodedlifeData[cellId] = dataBucket >> 7;
			resultDecodedlifeData[cellId + 1] = (dataBucket >> 6) & 0x01;
			resultDecodedlifeData[cellId + 2] = (dataBucket >> 5) & 0x01;
			resultDecodedlifeData[cellId + 3] = (dataBucket >> 4) & 0x01;
			resultDecodedlifeData[cellId + 4] = (dataBucket >> 3) & 0x01;
			resultDecodedlifeData[cellId + 5] = (dataBucket >> 2) & 0x01;
			resultDecodedlifeData[cellId + 6] = (dataBucket >> 1) & 0x01;
			resultDecodedlifeData[cellId + 7] = dataBucket & 0x01;
		}

	}


	/// Runs a kernel that decodes data from bit-per-cell to byte-per-cell format.
	void runBitLifeDecodeKernel(const ubyte* d_encodedLife, uint worldWidth, uint worldHeight, ubyte* d_lifeData) {

		assert(worldWidth % 8 == 0);
		uint worldEncDataWidth = worldWidth / 8;
		uint encWorldSize = worldEncDataWidth * worldHeight;

		ushort threadsCount = 256;
		assert(encWorldSize % threadsCount == 0);
		uint reqBlocksCount = encWorldSize / threadsCount;
		ushort blocksCount = ushort(std::min(32768u, reqBlocksCount));

		// decode life data back to byte per cell format
		bitLifeDecodeKernel<<<blocksCount, threadsCount>>>(d_encodedLife, encWorldSize, d_lifeData);
		checkCudaErrors(cudaDeviceSynchronize());
}


	/// CUDA device function that evaluates state of lookup table based on coordinates and key (state).
	__device__ inline uint getCellState(uint x, uint y, uint key) {
		uint index = y * 6 + x;
		return (key >> ((3 * 6 - 1) - index)) & 0x1;
	}

	/// CUDA kernel that computes the 6x3 lookup table.
	/// Needs to be invoked for each entry in lookup table (table size is 2^(6 * 3)).
	__global__ void precompute6x3EvaluationTableKernel(ubyte* resultEvalTableData) {

		uint tableIndex = blockIdx.x * blockDim.x + threadIdx.x;
		ubyte resultState = 0;

		// For each cell.
		for (uint dx = 0; dx < 4; ++dx) {
			// Count alive neighbors.
			uint aliveCount = 0;
			for (uint x = 0; x < 3; ++x) {
				for (uint y = 0; y < 3; ++y) {
					aliveCount += getCellState(x + dx, y, tableIndex);
				}
			}

			uint centerState = getCellState(1 + dx, 1, tableIndex);
			aliveCount -= centerState;  // Do not count center cell in the sum.

			if (aliveCount == 3 || (aliveCount == 2 && centerState == 1)) {
				resultState |= 1 << (3 - dx);
			}
		}

		resultEvalTableData[tableIndex] = resultState;
	}

	/// Runs a kernel that computes the 6x3 lookup table.
	void runPrecompute6x3EvaluationTableKernel(ubyte* d_lookupTable) {
		size_t lookupTableSize = 1 << (6 * 3);
		ushort threadsCount = 256;
		assert(lookupTableSize % threadsCount == 0);
		size_t reqBlocksCount = lookupTableSize / threadsCount;
		assert(reqBlocksCount < 65536);
		ushort blocksCount = (ushort)reqBlocksCount;

		precompute6x3EvaluationTableKernel<<<blocksCount, threadsCount>>>(d_lookupTable);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/// CUDA kernel that evaluates bit-per-cell life world using lookup table.
	/// Needs to be called (worldDataWidth * worldHeight) / bytesPerThread times.
	/// Note that worldDataWidth % bytesPerThread must be 0 (threads can not jump between rows).
	///
	/// @param lifeData  Linearized 2D array of life data with bit-per-cell density.
	/// @param worldDataWidth  Width of life data in bytes (width / 8).
	/// @param worldHeight  Height of life data (same as worldDataHeight would be).
	/// @param bytesPerThread  Number of bytes of life data processed per thread.
	/// @param evalTableData  Evaluation lookup table 6 x 3 (for 4 bits of data).
	/// @param resultLifeData  Result buffer in the same format as input.
	__global__ void bitLifeKernelLookup(const ubyte* lifeData, uint worldDataWidth, uint worldHeight,
			uint bytesPerThread, const ubyte* evalTableData, ubyte* resultLifeData) {

		uint worldSize = (worldDataWidth * worldHeight);

		for (uint cellId = (blockIdx.x * blockDim.x + threadIdx.x) * bytesPerThread;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x * bytesPerThread) {

			uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
			uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
			uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
			uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

			// Initialize data with previous byte and current byte.
			uint data0 = (uint)lifeData[x + yAbsUp] << 8;
			uint data1 = (uint)lifeData[x + yAbs] << 8;
			uint data2 = (uint)lifeData[x + yAbsDown] << 8;

			x = (x + 1) % worldDataWidth;

			data0 |= (uint)lifeData[x + yAbsUp];
			data1 |= (uint)lifeData[x + yAbs];
			data2 |= (uint)lifeData[x + yAbsDown];

			for (uint i = 0; i < bytesPerThread; ++i) {
				uint oldX = x;  // Old x is referring to current center cell.
				x = (x + 1) % worldDataWidth;
				data0 = (data0 << 8) | (uint)lifeData[x + yAbsUp];
				data1 = (data1 << 8) | (uint)lifeData[x + yAbs];
				data2 = (data2 << 8) | (uint)lifeData[x + yAbsDown];

				uint lifeStateHi = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
				uint lifeStateLo = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

				resultLifeData[oldX + yAbs] = (evalTableData[lifeStateHi] << 4) | evalTableData[lifeStateLo];
			}
		}
	}

	/// CUDA kernel that evaluates bit-per-cell life world using alive cells counting.
	/// Parameters are the same as @see bitLifeKernelLookup.
	__global__ void bitLifeKernelCounting(const ubyte* lifeData, uint worldDataWidth, uint worldHeight,
			uint bytesPerThread, ubyte* resultLifeData) {

		uint worldSize = (worldDataWidth * worldHeight);

		for (uint cellId = (blockIdx.x * blockDim.x + threadIdx.x) * bytesPerThread;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x * bytesPerThread) {

			uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
			uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
			uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
			uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

			// Initialize data with previous byte and current byte.
			uint data0 = (uint)lifeData[x + yAbsUp] << 16;
			uint data1 = (uint)lifeData[x + yAbs] << 16;
			uint data2 = (uint)lifeData[x + yAbsDown] << 16;

			x = (x + 1) % worldDataWidth;
			data0 |= (uint)lifeData[x + yAbsUp] << 8;
			data1 |= (uint)lifeData[x + yAbs] << 8;
			data2 |= (uint)lifeData[x + yAbsDown] << 8;

			for (uint i = 0; i < bytesPerThread; ++i) {
				uint oldX = x;  // Old x is referring to current center cell.
				x = (x + 1) % worldDataWidth;
				data0 |= (uint)lifeData[x + yAbsUp];
				data1 |= (uint)lifeData[x + yAbs];
				data2 |= (uint)lifeData[x + yAbsDown];

				uint result = 0;
				for (uint j = 0; j < 8; ++j) {
					// 23 ops.
					//uint aliveCells = ((data0 >> 14) & 0x1u) + ((data0 >> 15) & 0x1u) + ((data0 >> 16) & 0x1u)
					//	+ ((data1 >> 14) & 0x1) + ((data1 >> 16) & 0x1)  // Do not count middle cell.
					//	+ ((data2 >> 14) & 0x1u) + ((data2 >> 15) & 0x1u) + ((data2 >> 16) & 0x1u);

					// 10 ops + modulo.
					//unsigned long long state = unsigned long long(((data0 & 0x1C000) >> 8)
					//	| ((data1 & 0x14000) >> 11) | ((data2 & 0x1C000) >> 14));
					//assert(sizeof(state) == 8);
					//uint aliveCells = uint((state * 0x200040008001ULL & 0x111111111111111ULL) % 0xf);

					// 15 ops
					uint aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
					aliveCells >>= 14;
					aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
						+ ((data2 >> 15) & 0x1u);

					result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

					data0 <<= 1;
					data1 <<= 1;
					data2 <<= 1;
				}

				resultLifeData[oldX + yAbs] = result;
			}
		}
	}

	/// CUDA device function that swaps endianess of a 32 bits word.
	__device__ inline uint swapEndianessUint32(uint val) {
		val = ((val << 8) & 0xFF00FF00u) | ((val >> 8) & 0xFF00FFu);
		return (val << 16) | ((val >> 16) & 0xFFFFu);
	}

	/// CUDA kernel that evaluates bit-per-cell life world using alive cells counting in longer words.
	/// Parameters are the same as @see bitLifeKernelLookup.
	__global__ void bitLifeKernelCountingBigChunks(const uint* lifeData, uint worldDataWidth, uint worldHeight,
			uint chunksPerThread, uint* resultLifeData) {

		uint worldSize = (worldDataWidth * worldHeight);

		for (uint cellId = (blockIdx.x * blockDim.x + threadIdx.x) * chunksPerThread;
				cellId < worldSize;
				cellId += blockDim.x * gridDim.x * chunksPerThread) {

			uint x = (cellId + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
			uint yAbs = (cellId / worldDataWidth) * worldDataWidth;
			uint yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
			uint yAbsDown = (yAbs + worldDataWidth) % worldSize;

			// All read data are in little endian form. Swap is needed to allow simple counting.
			uint currData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
			uint currData1 = swapEndianessUint32(lifeData[x + yAbs]);
			uint currData2 = swapEndianessUint32(lifeData[x + yAbsDown]);

			x = (x + 1) % worldDataWidth;

			uint nextData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
			uint nextData1 = swapEndianessUint32(lifeData[x + yAbs]);
			uint nextData2 = swapEndianessUint32(lifeData[x + yAbsDown]);

			for (uint i = 0; i < chunksPerThread; ++i) {
				// Evaluate front overlapping cell.
				uint aliveCells = (currData0 & 0x1u) + (currData1 & 0x1u) + (currData2 & 0x1u)
					+ (nextData0 >> 31) + (nextData2 >> 31)  // Do not count middle cell.
					+ ((nextData0 >> 30) & 0x1u) + ((nextData1 >> 30) & 0x1u) + ((nextData2 >> 30) & 0x1u);

				// 31-st bit.
				uint result = (aliveCells == 3 || (aliveCells == 2 && (nextData1 >> 31))) ? (1u << 31) : 0u;

				uint oldX = x;  // Old x is referring to current center cell.
				x = (x + 1) % worldDataWidth;
				currData0 = nextData0;
				currData1 = nextData1;
				currData2 = nextData2;

				nextData0 = swapEndianessUint32(lifeData[x + yAbsUp]);
				nextData1 = swapEndianessUint32(lifeData[x + yAbs]);
				nextData2 = swapEndianessUint32(lifeData[x + yAbsDown]);

				// Evaluate back overlapping cell.
				aliveCells = ((currData0 >> 1) & 0x1u) + ((currData1 >> 1) & 0x1u) + ((currData2 >> 1) & 0x1u)
					+ (currData0 & 0x1u) + (currData2 & 0x1u)  // Do not count middle cell.
					+ (nextData0 >> 31) + (nextData1 >> 31) + (nextData2 >> 31);

				// 0-th bit.
				result |= (aliveCells == 3 || (aliveCells == 2 && (currData1 & 0x1u))) ? 1u : 0u;

				// The middle cells with no overlap.
				for (uint j = 0; j < 30; ++j) {
					uint shiftedData = currData0 >> j;
					uint aliveCells = (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u) + ((shiftedData >> 2) & 0x1u);

					shiftedData = currData2 >> j;
					aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u) + ((shiftedData >> 2) & 0x1u);

					shiftedData = currData1 >> j;
					aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 2) & 0x1u);  // Do not count middle cell.

					result |= (aliveCells == 3 || (aliveCells == 2 && (shiftedData & 0x2)) ? (2u << j) : 0u);
				}

				// Final swap from big to little endian form on the result.
				resultLifeData[oldX + yAbs] = swapEndianessUint32(result);
			}
		}
	}

	/// Runs a kernel that evaluates given world of bit-per-cell density using algorithm specified by parameters.
	bool runBitLifeKernel(ubyte*& d_encodedLifeData, ubyte*& d_encodedlifeDataBuffer, const ubyte* d_lookupTable,
			size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, uint bytesPerThread,
			bool useBigChunks) {

		// World has to fit into 8 bits of every byte exactly.
		if (worldWidth % 8 != 0) {
			return false;
		}

		size_t worldEncDataWidth = worldWidth / 8;
		if (d_lookupTable == nullptr && useBigChunks) {
			size_t factor = sizeof(uint) / sizeof(ubyte);
			if (factor != 4) {
				return false;
			}

			if (worldEncDataWidth % factor != 0) {
				return false;
			}
			worldEncDataWidth /= factor;
		}

		if (worldEncDataWidth % bytesPerThread != 0) {
			return false;
		}

		size_t encWorldSize = worldEncDataWidth * worldHeight;
		if (encWorldSize > std::numeric_limits<uint>::max()) {
			// TODO: fix kernels to work with world bit sizes.
			return false;
		}

		if ((encWorldSize / bytesPerThread) % threadsCount != 0) {
			return false;
		}

		size_t reqBlocksCount = (encWorldSize / bytesPerThread) / threadsCount;
		ushort blocksCount = ushort(std::min(size_t(32768), reqBlocksCount));

		if (d_lookupTable == nullptr) {
			if (useBigChunks) {
				// Does this really work?! Apparently yes.
				uint*& data = (uint*&)d_encodedLifeData;
				uint*& result = (uint*&)d_encodedlifeDataBuffer;

				for (size_t i = 0; i < iterationsCount; ++i) {
					bitLifeKernelCountingBigChunks<<<blocksCount, threadsCount>>>(data, uint(worldEncDataWidth),
						uint(worldHeight), bytesPerThread, result);
					std::swap(data, result);
				}
			}
			else {
				for (size_t i = 0; i < iterationsCount; ++i) {
					bitLifeKernelCounting<<<blocksCount, threadsCount>>>(d_encodedLifeData, uint(worldEncDataWidth),
						uint(worldHeight), bytesPerThread, d_encodedlifeDataBuffer);
					std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
				}
			}
		}
		else {
			for (size_t i = 0; i < iterationsCount; ++i) {
				bitLifeKernelLookup<<<blocksCount, threadsCount>>>(d_encodedLifeData, uint(worldEncDataWidth),
					uint(worldHeight), bytesPerThread, d_lookupTable, d_encodedlifeDataBuffer);
				std::swap(d_encodedLifeData, d_encodedlifeDataBuffer);
			}
		}

		checkCudaErrors(cudaDeviceSynchronize());
		return true;
	}

}