#pragma once

namespace mf {

	template<typename NoCppFileNeeded = int>
	class TCpuLife {

	protected:
		/// Main data for byte-per-pixel world.
		ubyte* m_data;
		/// Helper buffer for byte-per-pixel world.
		ubyte* m_resultData;

		/// Main data for bit-per-pixel world.
		/// Also used as buffer for rendering since it needs bit-per-pixel data.
		ubyte* m_bpcData;
		/// Helper buffer for bit-per-pixel world.
		ubyte* m_bpcResultData;

		/// Lookup table that evaluated any 6x3 area (9-bit key) to 4 bits of results.
		/// Every row uses only first four bits, last four bits are zeros.
		ubyte* m_lookupTable;

		/// Current width of world.
		size_t m_worldWidth;
		/// Current height of world.
		size_t m_worldHeight;
		/// Current data length (product of width and height)
		size_t m_dataLength;  // m_worldWidth * m_worldHeight

		/// Good random generator for world initialization.
		std::mt19937 m_randGen;


	public:
		TCpuLife()
				: m_data(nullptr)
				, m_resultData(nullptr)
				, m_bpcData(nullptr)
				, m_bpcResultData(nullptr)
				, m_lookupTable(nullptr)
				, m_worldWidth(0)
				, m_worldHeight(0)
				, m_dataLength(0) {

			// Initialize crappy built-in random generators.
			srand(uint(time(NULL)));

			// Properly initialize mt19937 random generator.
			std::array<int, std::mt19937::state_size> seed_data;
			std::random_device r;
			std::generate_n(seed_data.data(), seed_data.size(), std::ref(r));
			std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
			m_randGen = std::mt19937(seq);

			precompute6x3EvaluationTable();
		}

		~TCpuLife() {
			delete[] m_lookupTable;
			m_lookupTable = nullptr;

			freeBuffers();
		}

		/// Returns width of current world in cells.
		size_t getWorldWidth() const {
			return m_worldWidth;
		}

		/// Returns height of current world in cells.
		size_t getWorldHeight() const {
			return m_worldHeight;
		}

		/// Returns const byte-per-cell data.
		const ubyte* getLifeData() const {
			return m_data;
		}

		/// Returns byte-per-cell data.
		ubyte* lifeData() {
			return m_data;
		}

		/// Returns const bit-per-cell data.
		const ubyte* getBpcLifeData() const {
			return m_bpcData;
		}

		/// Returns bit-per-cell data.
		ubyte* bpcLifeData() {
			return m_bpcData;
		}

		/// Returns 6x3 to 4x1 evaluation table with 2^18 entries.
		const ubyte* getLookupTable() const {
			return m_lookupTable;
		}

		/// Returns true if buffers for given life algorithm type are allocated and ready for use.
		bool areBuffersAllocated(bool bitLife) const {
			if (bitLife) {
				return m_bpcData != nullptr && m_bpcResultData != nullptr;
			}
			else {
				return m_data != nullptr && m_resultData != nullptr && m_bpcData != nullptr;
			}
		}

		/// Frees all dynamically allocated buffers (expect lookup table).
		void freeBuffers() {
			delete[] m_data;
			m_data = nullptr;

			delete[] m_resultData;
			m_resultData = nullptr;

			delete[] m_bpcData;
			m_bpcData = nullptr;

			delete[] m_bpcResultData;
			m_bpcResultData = nullptr;

			m_dataLength = 0;
			// Do not free lookup table.
		}

		/// Frees all buffers and allocated buffers necessary for given algorithm type.
		bool allocBuffers(bool bitLife) {
			freeBuffers();

			size_t dataLength = m_worldWidth * m_worldHeight;
			assert(dataLength % 8 == 0);
			size_t bitDataLength = dataLength / 8;

			try {
				// Bit-per-cell buffer is always needed for display.
				m_bpcData = new ubyte[bitDataLength];

				if (bitLife) {
					m_dataLength = bitDataLength;
					m_bpcResultData = new ubyte[m_dataLength];
				}
				else {
					m_dataLength = dataLength;
					m_data = new ubyte[m_dataLength];
					m_resultData = new ubyte[m_dataLength];
				}
			}
			catch (std::bad_alloc&) {
				freeBuffers();
				return false;
			}

			return true;
		}

		/// Resizes the world and frees old buffers.
		/// Do not allocates new buffers (lazy allocation, buffers are allocated when needed).
		void resize(size_t newWidth, size_t newHeight) {
			freeBuffers();

			m_worldWidth = newWidth;
			m_worldHeight = newHeight;
		}

		/// Initializes CPU buffers (automatically recognizes life algorithm type).
		void initThis(bool useBetterRandom) {
			if (m_data != nullptr) {
				// Normal life.
				init(m_data, m_dataLength, 0x1u, useBetterRandom);
			}
			else {
				// Bit life.
				init(m_bpcData, m_dataLength, 0xFFu, useBetterRandom);
			}
		}

		/// Initializes given buffer with random data using on normal (crappy rand()) or better (std::mt19937) random
		/// generator. The mask is applied to random number before saving to the element. Use mask 0x1 to byte-per-cell
		/// and mask 0xFF to bit-per-cell.
		void init(ubyte* data, size_t length, uint mask, bool useBetterRandom) {
			if (useBetterRandom) {
				for (size_t i = 0; i < length; ++i) {
					data[i] = ubyte(m_randGen() & mask);
				}
			}
			else {
				for (size_t i = 0; i < length; ++i) {
					data[i] = ubyte(rand() & mask);
				}
			}
		}

		/// Encodes internal byte-per-cell data buffer to internal to bit-per-cell data buffer.
		/// Used for for display.
		void encodeDataToBpc() {
			assert(m_data != nullptr);

			ubyte* data = m_data;
			ubyte* encData = m_bpcData;
			size_t dataLength = m_worldWidth *  m_worldHeight;
			size_t encDataLength = dataLength / 8;
			std::memset(encData, 0, encDataLength);
			for (size_t i = 0; i < dataLength; ++i) {
				encData[i / 8] |= data[i] << (7 - (i % 8));
			}
		}

		/// Evaluates life based on given parameters.
		bool iterate(size_t lifeIteratinos, bool parallel, bool useLambdaInParallel, bool bitLife,
				size_t blocksPerThread, bool useLookup, bool useBigChunks) {

			if (parallel || bitLife) {
				if (bitLife) {
					return iterateBitLife(lifeIteratinos, blocksPerThread, useLookup, useBigChunks);
				}
				else {
					if (useLambdaInParallel) {
						return cpuLife.iterateParallelLambda(lifeIteratinos);
					}
					else {
						return cpuLife.iterateParallelStaticFunc(lifeIteratinos);
					}
				}
			}
			else {
				return cpuLife.iterateSerial(lifeIteratinos);
			}
		}

		/// Serial version of standard byte-per-cell life.
		bool iterateSerial(size_t iterations) {
			for (size_t i = 0; i < iterations; ++i) {
				for (size_t y = 0; y < m_worldHeight; ++y) {
					size_t y0 = ((y + m_worldHeight - 1) % m_worldHeight) * m_worldWidth;
					size_t y1 = y * m_worldWidth;
					size_t y2 = ((y + 1) % m_worldHeight) * m_worldWidth;

					for (size_t x = 0; x < m_worldWidth; ++x) {
						size_t x0 = (x + m_worldWidth - 1) % m_worldWidth;
						size_t x2 = (x + 1) % m_worldWidth;

						ubyte aliveCells = countAliveCells(m_data, x0, x, x2, y0, y1, y2);
						m_resultData[y1 + x] = aliveCells == 3 || (aliveCells == 2 && m_data[x + y1]) ? 1 : 0;
					}
				}

				std::swap(m_data, m_resultData);
			}

			return true;
		}

		/// Parallel version of standard byte-per-cell life (using lambda function).
		bool iterateParallelLambda(size_t iterations) {
			auto evaluateCell = [&] (size_t index) {
				size_t x1 = index % m_worldWidth;
				size_t y1 = index - x1;

				size_t y0 = (y1 + m_dataLength - m_worldWidth) % m_dataLength;
				size_t y2 = (y1 + m_worldWidth) % m_dataLength;

				size_t x0 = (x1 + m_worldWidth - 1) % m_worldWidth;
				size_t x2 = (x1 + 1) % m_worldWidth;

				ubyte aliveCells = countAliveCells(m_data, x0, x1, x2, y0, y1, y2);

				m_resultData[y1 + x1] = aliveCells == 3 || (aliveCells == 2 && m_data[x1 + y1]) ? 1 : 0;
			};

			for (size_t i = 0; i < iterations; ++i) {
				Concurrency::parallel_for<size_t>(0, m_dataLength, 1, evaluateCell);
				std::swap(m_data, m_resultData);
			}

			return true;
		}

		/// Parallel version of standard byte-per-cell life (using static function).
		bool iterateParallelStaticFunc(size_t iterations) {
			s_data = m_data;
			s_resultData = m_resultData;
			s_worldWidth = m_worldWidth;
			s_dataLength = m_dataLength;

			for (size_t i = 0; i < iterations; ++i) {
				Concurrency::parallel_for<size_t>(0, m_dataLength, 1, &iterateParallelStep);
				std::swap(s_data, s_resultData);
			}

			m_data = s_data;
			m_resultData = s_resultData;
			return true;
		}

		/// Evaluates bit-per-cell life based on given parameters.
		bool iterateBitLife(size_t iterations, size_t blocksPerThread, bool useLookup, bool useBigBlocks) {
			if (m_worldWidth % 8 != 0) {
				return false;
			}

			size_t worldEncDataWidth = m_worldWidth / 8;
			if (!useLookup && useBigBlocks) {
				size_t factor = sizeof(uint) / sizeof(ubyte);
				if (factor != 4) {
					return false;
				}

				if (worldEncDataWidth % factor != 0) {
					return false;
				}

				worldEncDataWidth /= factor;
			}

			size_t encWorldSize = worldEncDataWidth * m_worldHeight;

			if (worldEncDataWidth % blocksPerThread != 0) {
				return false;
			}

			if (useLookup) {
				return iterateBitPerPixelLookupParallelLambda(iterations, blocksPerThread, worldEncDataWidth,
					encWorldSize);
			}
			else {
				if (useBigBlocks) {
					return iterateBitPerPixelCountingBigParallelLambda(iterations, blocksPerThread, worldEncDataWidth,
						encWorldSize);
				}
				else {
					return iterateBitPerPixelCountingParallelLambda(iterations, blocksPerThread, worldEncDataWidth,
						encWorldSize);
				}
			}
		}

	private:
		/// Parallel version of bit-per-cell life using lookup table.
		bool iterateBitPerPixelLookupParallelLambda(size_t iterations, size_t blocksPerThread, size_t worldDataWidth,
				size_t worldSize) {
			assert(m_lookupTable != nullptr);

			auto evaluateCell = [worldDataWidth, worldSize, blocksPerThread, this] (size_t index) {
				index *= blocksPerThread;
				size_t x = (index + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
				size_t yAbs = (index / worldDataWidth) * worldDataWidth;
				size_t yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
				size_t yAbsDown = (yAbs + worldDataWidth) % worldSize;

				// Initialize data with previous byte and current byte.
				uint data0 = uint(m_bpcData[x + yAbsUp]) << 8;
				uint data1 = uint(m_bpcData[x + yAbs]) << 8;
				uint data2 = uint(m_bpcData[x + yAbsDown]) << 8;

				x = (x + 1) % worldDataWidth;

				data0 |= uint(m_bpcData[x + yAbsUp]);
				data1 |= uint(m_bpcData[x + yAbs]);
				data2 |= uint(m_bpcData[x + yAbsDown]);

				for (size_t i = 0; i < blocksPerThread; ++i) {
					size_t oldX = x;  // Old x is referring to current center cell.
					x = (x + 1) % worldDataWidth;
					data0 = (data0 << 8) | uint(m_bpcData[x + yAbsUp]);
					data1 = (data1 << 8) | uint(m_bpcData[x + yAbs]);
					data2 = (data2 << 8) | uint(m_bpcData[x + yAbsDown]);

					uint lifeStateHi = ((data0 & 0x1F800) << 1) | ((data1 & 0x1F800) >> 5) | ((data2 & 0x1F800) >> 11);
					uint lifeStateLo = ((data0 & 0x1F80) << 5) | ((data1 & 0x1F80) >> 1) | ((data2 & 0x1F80) >> 7);

					m_bpcResultData[oldX + yAbs] = (m_lookupTable[lifeStateHi] << 4) | m_lookupTable[lifeStateLo];
				}
			};

			for (size_t i = 0; i < iterations; ++i) {
				Concurrency::parallel_for<size_t>(0, worldSize / blocksPerThread, 1, evaluateCell);
				std::swap(m_bpcData, m_bpcResultData);
			}

			return true;
		}

		/// Parallel version of bit-per-cell life using bit counting.
		bool iterateBitPerPixelCountingParallelLambda(size_t iterations, size_t blocksPerThread, size_t worldDataWidth,
				size_t worldSize) {

			auto evaluateCell = [worldDataWidth, worldSize, blocksPerThread, this] (size_t index) {
				index *= blocksPerThread;
				size_t x = (index + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
				size_t yAbs = (index / worldDataWidth) * worldDataWidth;
				size_t yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
				size_t yAbsDown = (yAbs + worldDataWidth) % worldSize;

				// Initialize data with previous byte and current byte.
				uint data0 = (uint)m_bpcData[x + yAbsUp] << 16;
				uint data1 = (uint)m_bpcData[x + yAbs] << 16;
				uint data2 = (uint)m_bpcData[x + yAbsDown] << 16;

				x = (x + 1) % worldDataWidth;

				data0 |= (uint)m_bpcData[x + yAbsUp] << 8;
				data1 |= (uint)m_bpcData[x + yAbs] << 8;
				data2 |= (uint)m_bpcData[x + yAbsDown] << 8;

				for (uint i = 0; i < blocksPerThread; ++i) {
					size_t oldX = x;  // Old x is referring to current center cell.
					x = (x + 1) % worldDataWidth;
					data0 |= (uint)m_bpcData[x + yAbsUp];
					data1 |= (uint)m_bpcData[x + yAbs];
					data2 |= (uint)m_bpcData[x + yAbsDown];

					uint result = 0;
					for (uint j = 0; j < 8; ++j) {
						// 23 ops.
						//uint aliveCells = ((data0 >> 14) & 0x1u) + ((data0 >> 15) & 0x1u) + ((data0 >> 16) & 0x1u)
						//	+ ((data1 >> 14) & 0x1) + ((data1 >> 16) & 0x1)  // Do not count middle cell.
						//	+ ((data2 >> 14) & 0x1u) + ((data2 >> 15) & 0x1u) + ((data2 >> 16) & 0x1u);

						// 10 ops + modulo.
						//unsigned long long state = unsigned long long(((data0 & 0x1C000) >> 8)
						// | ((data1 & 0x14000) >> 11) | ((data2 & 0x1C000) >> 14));
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

					m_bpcResultData[oldX + yAbs] = ubyte(result);
				}
			};

			for (size_t i = 0; i < iterations; ++i) {
				Concurrency::parallel_for<size_t>(0, worldSize / blocksPerThread, 1, evaluateCell);
				std::swap(m_bpcData, m_bpcResultData);
			}

			return true;
		}

		/// Parallel version of bit-per-cell life using bit counting on big blocks.
		bool iterateBitPerPixelCountingBigParallelLambda(size_t iterations, size_t consecutiveBlocks,
				size_t worldDataWidth, size_t worldSize) {

			uint* dataPtr = (uint*)m_bpcData;
			uint* resultPtr = (uint*)m_bpcResultData;

			auto evaluateCell = [worldDataWidth, worldSize, dataPtr, resultPtr, consecutiveBlocks, this]
					(size_t index) {

				index *= consecutiveBlocks;
				size_t x = (index + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
				size_t yAbs = (index / worldDataWidth) * worldDataWidth;
				size_t yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
				size_t yAbsDown = (yAbs + worldDataWidth) % worldSize;

				uint currData0 = swapEndianessUint32(dataPtr[x + yAbsUp]);
				uint currData1 = swapEndianessUint32(dataPtr[x + yAbs]);
				uint currData2 = swapEndianessUint32(dataPtr[x + yAbsDown]);

				x = (x + 1) % worldDataWidth;
				uint nextData0 = swapEndianessUint32(dataPtr[x + yAbsUp]);
				uint nextData1 = swapEndianessUint32(dataPtr[x + yAbs]);
				uint nextData2 = swapEndianessUint32(dataPtr[x + yAbsDown]);

				for (uint i = 0; i < consecutiveBlocks; ++i) {
					// Evaluate front overlapping cell.
					uint aliveCells = (currData0 & 0x1u) + (currData1 & 0x1u) + (currData2 & 0x1u)
						+ (nextData0 >> 31) + (nextData2 >> 31)  // Do not count middle cell.
						+ ((nextData0 >> 30) & 0x1u) + ((nextData1 >> 30) & 0x1u) + ((nextData2 >> 30) & 0x1u);

					// 31-st bit.
					uint result = (aliveCells == 3 || (aliveCells == 2 && (nextData1 >> 31))) ? (1u << 31) : 0u;

					size_t oldX = x;  // Old x is referring to current center cell.
					x = (x + 1) % worldDataWidth;
					currData0 = nextData0;
					currData1 = nextData1;
					currData2 = nextData2;

					nextData0 = swapEndianessUint32(dataPtr[x + yAbsUp]);
					nextData1 = swapEndianessUint32(dataPtr[x + yAbs]);
					nextData2 = swapEndianessUint32(dataPtr[x + yAbsDown]);

					// Evaluate back overlapping cell.
					aliveCells = ((currData0 >> 1) & 0x1u) + ((currData1 >> 1) & 0x1u) + ((currData2 >> 1) & 0x1u)
						+ (currData0 & 0x1u) + (currData2 & 0x1u)  // Do not count middle cell.
						+ (nextData0 >> 31) + (nextData1 >> 31) + (nextData2 >> 31);

					// 0-th bit.
					result |= (aliveCells == 3 || (aliveCells == 2 && (currData1 & 0x1u))) ? 1u : 0u;

					// The middle cells with no overlap.
					for (uint j = 0; j < 30; ++j) {
						uint shiftedData = currData0 >> j;
						uint aliveCells = (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u)
							+ ((shiftedData >> 2) & 0x1u);

						shiftedData = currData2 >> j;
						aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 1) & 0x1u) + ((shiftedData >> 2) & 0x1u);

						shiftedData = currData1 >> j;
						aliveCells += (shiftedData & 0x1u) + ((shiftedData >> 2) & 0x1u);  // Do not count middle cell.

						result |= (aliveCells == 3 || (aliveCells == 2 && (shiftedData & 0x2)) ? (2u << j) : 0u);
					}

					resultPtr[oldX + yAbs] = swapEndianessUint32(result);
				}
			};

			for (size_t i = 0; i < iterations; ++i) {
				Concurrency::parallel_for<size_t>(0, worldSize / consecutiveBlocks, 1, evaluateCell);
				std::swap(dataPtr, resultPtr);
				std::swap(m_bpcData, m_bpcResultData);
			}

			return true;
		}


	private:
		// Static variables for static function usage only.
		static ubyte* s_data;
		static ubyte* s_resultData;
		static size_t s_worldWidth;
		static size_t s_dataLength;

		/// Counts alive cells in given data on given coords.
		/// Y-coordinates y0, y1 and y2 are already pre-multiplied with world width.
		static inline ubyte countAliveCells(ubyte* data, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1,
				size_t y2) {

			return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
				+ data[x0 + y1] + data[x2 + y1]
				+ data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
		}

		/// Static function for parallel life evaluation.
		static void iterateParallelStep(size_t index) {
			size_t x1 = index % s_worldWidth;
			size_t y1 = index - x1;

			size_t y0 = (y1 + s_dataLength - s_worldWidth) % s_dataLength;
			size_t y2 = (y1 + s_worldWidth) % s_dataLength;

			size_t x0 = (x1 + s_worldWidth - 1) % s_worldWidth;
			size_t x2 = (x1 + 1) % s_worldWidth;

			ubyte aliveCells = countAliveCells(s_data, x0, x1, x2, y0, y1, y2);

			s_resultData[y1 + x1] = aliveCells == 3 || (aliveCells == 2 && s_data[x1 + y1]) ? 1 : 0;
		}

		/// Swaps endianess in 32 bit word.
		inline static uint swapEndianessUint32(uint val) {
			val = ((val << 8) & 0xFF00FF00u) | ((val >> 8) & 0xFF00FFu);
			return (val << 16) | ((val >> 16) & 0xFFFFu);
		}

		/// Evaluates state of lookup table based on coordinates and key (state).
		inline static size_t getLookupTableCellState(size_t x, size_t y, size_t key) {
			size_t index = y * 6 + x;
			return (key >> ((3 * 6 - 1) - index)) & 0x1u;
		}

		/// Computes the 6x3 lookup table.
		void precompute6x3EvaluationTable() {
			if (m_lookupTable != nullptr) {
				return;
			}

			auto evaluateTable = [&] (size_t index) {
				ubyte resultState = 0;

				// For each cell.
				for (size_t dx = 0; dx < 4; ++dx) {
					// Count alive neighbors.
					size_t aliveCount = 0;
					for (size_t x = 0; x < 3; ++x) {
						for (size_t y = 0; y < 3; ++y) {
							aliveCount += getLookupTableCellState(x + dx, y, index);
						}
					}

					size_t centerState = getLookupTableCellState(1 + dx, 1, index);
					aliveCount -= centerState;  // Do not count center cell in the sum.

					if (aliveCount == 3 || (aliveCount == 2 && centerState == 1)) {
						resultState |= 1 << (3 - dx);
					}
				}

				m_lookupTable[index] = resultState;
			};

			m_lookupTable = new ubyte[1 << 18];
			Concurrency::parallel_for<size_t>(0, 1 << 18, 1, evaluateTable);
		}

	};

	template <typename T> ubyte* TCpuLife<T>::s_data;
	template <typename T> ubyte* TCpuLife<T>::s_resultData;
	template <typename T> size_t TCpuLife<T>::s_worldWidth;
	template <typename T> size_t TCpuLife<T>::s_dataLength;

	typedef TCpuLife<> CpuLife;

}
