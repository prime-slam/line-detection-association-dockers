# Kernel-Based Hough Transform for Detecting Straight Lines in Images

This repository contains the reference implementation of the Kernel-Based Hough Transform (KHT). The KHT is a real-time line detection procedure that extends the conventional voting procedure of the [Hough transform](https://en.wikipedia.org/wiki/Hough_transform). It operates on clusters of approximately collinear pixels. For each cluster, the KHT casts votes using an oriented elliptical-Gaussian kernel that models the uncertainty associated with the best-fitting line for the corresponding cluster. The proposed approach not only significantly improves the performance of the voting scheme, but also produces a much cleaner voting map and makes the transform more robust to the detection of spurious lines.

Please cite our Pattern Recognition paper if you use this code in your research:

```{.bib}
@Article{fernandes_oliveira-pr-41(1)-2008,
    author  = {Fernandes, Leandro A. F. and Oliveira, Manuel M.},
    title   = {Real-time line detection through an improved {H}ough transform voting scheme},
    journal = {Pattern Recognition},
    year    = {2008},
    volume  = {41},
    number  = {1},
    pages   = {299--314},
    doi     = {https://doi.org/10.1016/j.patcog.2007.04.003},
    url     = {http://www.ic.uff.br/~laffernandes/projects/kht},
}
```

The paper presents a complete description of the implemented technique. You will find additional material on the [project's page](http://www.ic.uff.br/~laffernandes/projects/kht).

Please, let Leandro A. F. Fernandes ([laffernandes@ic.uff.br](mailto:laffernandes@ic.uff.br)) knows if you want to contribute to this project. Also, do not hesitate to contact him if you encounter any problems.

**Contents:**

1. [Requirements](#1-requirements)
2. [How to Install KHT](#2-how-to-install-kht)
3. [Compiling Examples](#3-compiling-examples)
4. [Related Project](#4-related-project)
5. [License](#5-license)
6. [Releases](#6-releases)

## 1. Requirements

Make sure that you have the following tools before attempting to use KHT.

Required tool:

- Your favorite [C++11](https://en.wikipedia.org/wiki/C%2B%2B11) compiler.
- [CMake](https://cmake.org) (version >= 3.14) to automate installation and to build and run examples.

Optional tool:

- [Python](https://www.python.org) 2 or 3 interpreter, if you want to build and use KHT with Python.
- [MATLAB](https://www.mathworks.com/products/matlab.html) (version >= R2007a), if you want to build and use KHT with MATLAB.
- [Virtual enviroment](https://wiki.archlinux.org/index.php/Python/Virtual_environment) to create an isolated workspace for a Python application.

The reference implementation of the KHT doesn't have any dependencies other than the [C++ standard library](https://en.cppreference.com/w/cpp/header). But some libraries are required if you want to use KHT with Python or build the sample Python and C++ applications.

Required Python packages and C++ libraries, if you want to use KHT with Python:

- [NumPy](https://numpy.org), the fundamental package for scientific computing with Python.
- [Boost.Python](https://www.boost.org/doc/libs/release/libs/python/doc/html/index.html) (version >= 1.56), a C++ library which enables seamless interoperability between C++ and the Python programming language.
- [Boost.NumPy](https://www.boost.org/doc/libs/release/libs/python/doc/html/numpy/index.html), a C++ library that extends Boost.Python to NumPy.

Required C++ libraries and Python packages, if you want to build the C++ and Python sample application, respectively:

- [OpenCV](https://opencv.org) (version >= 2.2), a C++ library of programming functions mainly aimed at real-time computer vision.
- [OpenCV-Python](https://pypi.org/project/opencv-python), a wrapper package for OpenCV Python bindings.
- [Matplotlib](https://matplotlib.org), a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats.

## 2. How to Install KHT

Use the [git clone](https://git-scm.com/docs/git-clone) command to download the project, where `<kht-dir>` must be replaced by the directory in which you want to place KHT's source code, or removed `<kht-dir>` from the command line to download the project to the `./kht` directory:

```bash
git clone https://github.com/laffernandes/kht.git <kht-dir>
```

The basic steps for configuring, building, and installing the KHT library look like this:

```bash
cd <kht-dir>/cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install
```

Notice that you may use the `-G <generator-name>` option of CMake's command-line tool to choose the build system (*e.g.*, Unix makefiles, Visual Studio, etc.). Please, refer to [CMake's Help](https://cmake.org/cmake/help/latest/manual/cmake.1.html) for a complete description of how to use the CMake's command-line tool.

### Installing the Python Module

We provide a back-end to access KHT from a Python environment. In order to make it available, you have to build the `kht` module, after installing the KHT library, using CMake of `setuptools`. With `setuptools` is easier, just run the commands presented bellow:

```bash
cd <kht-dir>/python
python setup.py install
```

With CMake, run the following commands instead:

```bash
cd <kht-dir>/python
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install
```

It is important to emphasize that both Python 2 and 3 are supported. Please, refer to [CMake's documentation](https://cmake.org/cmake/help/latest/module/FindPython.html) for details about how CMake finds the Python interpreter, compiler, and development environment.

Finally, add `<cmake-install-prefix>/lib/kht/python/<python-version>` to the the `PYTHONPATH` environment variable. The `<cmake-install-prefix>` placeholder usually is `/usr/local` on Linux, and `C:/Program Files/KHT` or `C:/Program Files (x86)/KHT` on Windows. But it may change according to what was set in CMake. The `<python-version>` placeholder is the version of the Python interpreter found by CMake.

Set the `PYTHONPATH` variable by calling following command in Linux:

```bash
export PYTHONPATH="$PYTHONPATH:<cmake-install-prefix>/lib/kht/python/<python-version>"
```

But this action is not permanent. The new value of `PYTHONPATH` will be lost as soon as you close the terminal. A possible solution to make an environment variable persistent for a user's environment is to export the variable from the user's profile script:

  1. Open the current user's profile (the `~/.bash_profile` file) into a text editor.
  2. Add the export command for the `PYTHONPATH` environment variable at the end of this file.
  3. Save your changes.

Execute the following steps to set the `PYTHONPATH` in Windows:

  1. From the *Windows Explorer*, right click the *Computer* icon.
  2. Choose *Properties* from the context menu.
  3. Click the *Advanced system settings* link.
  4. Click *Environment Variables*. In the section *System Variables*, find the `PYTHONPATH` environment variable and select it. Click *Edit*. If the `PYTHONPATH` environment variable does not exist, click *New*.
  5. In the *Edit System Variable* (or *New System Variable*) window, specify the value of the `PYTHONPATH` environment variable to include `"<cmake-install-prefix>/lib/kht/python/<python-version>"`. Click *OK*. Close all remaining windows by clicking *OK*.
  6. Reopen yout Python environment.

## Installing the MATLAB Wrapper

We provide a back-end to access KHT from MATLAB. In order to make it available, you have to build the MEX-file, after installing the KHT library, using the commands presented bellow:

```bash
cd <kht-dir>/matlab
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install
```

Finally, open MATLAB and use the following commands to the instaled wrapper to MATLAB's search path:

```matlab
addpath('<cmake-install-prefix>/lib/kht/matlab')
savepath
```

The `<cmake-install-prefix>` placeholder usually is `/usr/local` on Linux, and `C:/Program Files/KHT` or `C:/Program Files (x86)/KHT` on Windows. But it may change according to what was set in CMake.

Sometimes CMake tells you that "Could NOT find Matlab (missing: Matlab_MEX_LIBRARY)" even when MATLAB is appropriately installed. In this case, you have to check whether the correct architecture (32- vs. 64-bit compiler) was selected for compiling the KHT library and its MATLAB wrapper. For instance, on Windows, instead of using CMake with the "Visual Studio 15 2017" generator, try to use the "Visual Studio 15 2017 Win64" generator and vice versa.

## 3. Compiling Examples

The basic steps for configuring and building the C++ example of the KHT look like this:

```bash
cd <kht-dir>/cpp/example
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Call `./main` to run the executable file produced on Linux, and `Release\main.exe` to run on Windows.

Recall that `<kht-dir>` is the directory in which you placed KHT's source code.

Use the files in the `<kht-dir>/cpp/example` directory as examples of how to configure and call the reference implementation of the KHT from your C++ program. For instance, after installation of the KHT library, CMake will find KHT using the command `find_package(KHT)` (see the [`<kht-dir>/cpp/example/CMakeLists.txt`](cpp/example/CMakeLists.txt) file and the [CMake documentation](https://cmake.org/cmake/help/latest/command/find_package.html) for details). Also, you will be able to use the `KHT_INCLUDE_DIRS` variable in the `CMakeList.txt` file of your program while defining the include directories of your C++ project or targets. In your source code, you have to use the `#include <kht/kht.hpp>` directive to include the contents of the standard header file and then call the `kht` procedure (see the [`<kht-dir>/cpp/example/kht_example.cpp`](cpp/example/kht_example.cpp) file).

Similarly, you will find examples of how to use the KHT library with Python and MATLAB in, respectively, the [`<kht-dir>/python/example`](python/example) and [`<kht-dir>/matlab/example`](matlab/example) directories.

## 4. Related Project

Please, visit the [project's page](http://www.ic.uff.br/~laffernandes/projects/kht) to find a list of related projects.

## 5. License

This software is licensed under the GNU General Public License v3.0. See the [`LICENSE`](LICENSE) file for details.

## 6. Releases

Version 2.0.0

- *REPOSITORY*: The project moved from [SourceForge](https://sourceforge.net/projects/khtsandbox) to [GitHub](https://github.com/laffernandes/kht).
- *BUILD SYSTEM*: CMake becomes the default build system of KHT. Platform-specific solutions will not be provided anymore.
- *PYTHON*: The Python back-end was included.

Version 1.0.4

- *BUG FIX*: In the `next()` function of `linking.cpp` file, there were no protections against accessing pixels outside the image limits. The author would like to thank Timo Knuutile for pointing out this problem.

Version 1.0.3

- *UPDATE*: The `kht_compile.m` file was ported to MATLAB R2011b, and a Microsoft Visual Studio 2010 project was included.

Version 1.0.2

- *BUG FIX*: In line 177 of peak_detection.cpp file, the function `compare_bins` was being forced to assume a non-standard calling convention when passed as an argument to `std::qsort()` function, preventing its compilation on Linux. The problem was fixed. The authors would like to thank Laurens Leeuwis for pointing out this problem.

Version 1.0.1

- *BUG FIX*: During the initialization of the accumulator, the last item of the lookup table defining the discrete `theta` values was not initialized. In such a case, the value should be `180-delta` degrees. As a result, detected lines having `theta = 180-delta` were getting a random angular value. The authors would like to thank Dave Wood for pointing out this problem.

Version 1.0.0

- *NEW*: Everything is new.
