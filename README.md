
Conway's Game of Life on GPU using CUDA
=========
This project compares performance of CPU and GPU in evaluation of famous Conway's Game of Life.
The performance was tested on three different implementations.
The most sophisticated version of the algorithm on GPU stores data in one bit-per-cell array and leads to speed-up of 480x compared to serial CPU algorithm.
The best implementation for CPU turned out to be lookup-table approach leading to 60x speedups over serial CPU.

**Author**: Marek Fiser &lt; code@marekfiser.cz &gt;

**Project page**: http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA


**License**: Public domain, see LICENSE.txt for details.

See other readme files for inluded libraries: FreeGlut, Glew, and Google Test.

Features
--------

* CPU and CUDA GPU implementations of Conway's Game of Life.
  * Three different implementations of both CPU and GPU algorithms.
* Automatic benchmark with export to CSV.
* Unit tests assuring corectness.
* Code is well structured and commented.
* Cool post-process visual effects.


Compiling and running
--------

In order to compile/run this application you probably need to have CUDA SDK installed and your NVIDIA graphics card needs to have CUDA Capability at least 2.0.
All other necessary DLLs are included in this package.

There is also compiled executable in the bin folder.

In order to run GPU benchmark, the time limit on kernels needs to be disabled or
very high (~50 seconds) because some kernel configurations take very long time.
Alternatively, you can change the values in benchmark.h file or press Shift+Q
to run "debug" version of benchmark that has lower limits for life world sizes.