/* Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
 *
 * author     : Fernandes, Leandro A. F.
 * e-mail     : laffernandes@ic.uff.br
 * home page  : http://www.ic.uff.br/~laffernandes
 * 
 * This file is part of the reference implementation of the Kernel-Based
 * Hough Transform (KHT). The complete description of the implemented
 * techinique can be found at:
 * 
 *     Leandro A. F. Fernandes, Manuel M. Oliveira
 *     Real-time line detection through an improved Hough transform
 *     voting scheme, Pattern Recognition (PR), Elsevier, 41:1, 2008,
 *     pp. 299-314.
 * 
 *     DOI.........: https://doi.org/10.1016/j.patcog.2007.04.003
 *     Project Page: http://www.ic.uff.br/~laffernandes/projects/kht
 *     Repository..: https://github.com/laffernandes/kht
 * 
 * KHT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * KHT is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
 * License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with KHT. If not, see <https://www.gnu.org/licenses/>.
 */

#include <cctype>
#include <set>
#include <string>
#include <mex.h>
#include <kht/kht.hpp>

// Set of numerical types.
static std::set<mxClassID> const NUMERIC_TYPES = {mxDOUBLE_CLASS, mxSINGLE_CLASS, mxINT8_CLASS, mxUINT8_CLASS, mxINT16_CLASS, mxUINT16_CLASS, mxINT32_CLASS, mxUINT32_CLASS, mxINT64_CLASS, mxUINT64_CLASS};

// Argument check function.
inline void check(bool condition, char const *errorid, char const *errormsg) {
    if (!condition) mexErrMsgIdAndTxt(errorid, errormsg);
}

// The MEX-file gateway function.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    check(nrhs % 2 == 1, "KHT:narginchk:notEnoughInputs", "Invalid number of input arguments.");

    // Read the input binary image (the image was transposed).
    check(mxGetClassID(prhs[0]) == mxUINT8_CLASS && mxGetNumberOfDimensions(prhs[0]) == 2 && mxGetNumberOfElements(prhs[0]) > 1, "KHT:arginchk:invalidType", "The first input argument must be a single-channel 8-bit binary image.");

    std::size_t image_width = static_cast<std::size_t>(mxGetM(prhs[0]));
    std::size_t image_height = static_cast<std::size_t>(mxGetN(prhs[0]));
    std::uint8_t *binary_image = static_cast<std::uint8_t*>(mxGetData(prhs[0]));

    // Read other input arguments.
    std::int32_t cluster_min_size = 10;
    std::double_t cluster_min_deviation = 2;
    std::double_t delta = 0.5;
    std::double_t kernel_min_height = 0.002;
    std::double_t n_sigmas = 2;

    for (int i = 1; i < nrhs; i += 2) {
        check(mxGetClassID(prhs[i]) == mxCHAR_CLASS, "KHT:arginchk:invalidType", "Invalid input argument name type.");
        check(NUMERIC_TYPES.find(mxGetClassID(prhs[i + 1])) != NUMERIC_TYPES.end() && mxGetNumberOfElements(prhs[i + 1]) == 1, "KHT:arginchk:invalidType", ("Invalid input type for '" + std::string(mxArrayToUTF8String(prhs[i])) + "'.").c_str());

        std::string name = mxArrayToUTF8String(prhs[i]);
        std::transform(name.begin(), name.end(), name.begin(), [](auto c) { return std::tolower(c); });

        if (name.compare("cluster_min_size") == 0) cluster_min_size = static_cast<std::int32_t>(mxGetScalar(prhs[i + 1]));
        else if (name.compare("cluster_min_deviation") == 0) cluster_min_deviation = mxGetScalar(prhs[i + 1]);
        else if (name.compare("delta") == 0) delta = mxGetScalar(prhs[i + 1]);
        else if (name.compare("kernel_min_height") == 0) kernel_min_height = mxGetScalar(prhs[i + 1]);
        else if (name.compare("n_sigmas") == 0) n_sigmas = mxGetScalar(prhs[i + 1]);
        else check(false, "KHT:arginchk:unknown", ("Unknown argument '" + std::string(mxArrayToUTF8String(prhs[i])) + "'.").c_str());
    }

    // Execute the proposed Hough transform voting scheme and the peak detection procedure.
    static kht::ListOfLines lines;

    kht::run_kht(lines, binary_image, image_width, image_height, cluster_min_size, cluster_min_deviation, delta, kernel_min_height, n_sigmas);
    
    // Create and populate the resulting array of detected lines.
    plhs[0] = mxCreateDoubleMatrix(lines.size(), 2, mxREAL);
    std::double_t *dbuffer = mxGetPr(plhs[0]);

    for (std::size_t i = 0; i != lines.size(); ++i) {
        kht::Line &line = lines[i];
        dbuffer[i               ] = line.rho;
        dbuffer[i + lines.size()] = line.theta;
    }
}
