%KHT	Kernel-based Hough transform for detecting straight lines in images.
%   LINES = KHT(BINARY_IMAGE,...) is a wrapper function for the C++ implementation of the
%   improved Hough transform voting scheme proposed by Fernandes and Oliveira in
%
%      Leandro A. F. Fernandes, Manuel M. Oliveira
%      Real-time line detection through an improved Hough transform voting scheme
%      Pattern Recognition (PR), Elsevier, 41:1, 2008, pp. 299-314. <a href="matlab:web('https://doi.org/10.1016/j.patcog.2007.04.003')">DOI</a> <a href="matlab:web('http://www.ic.uff.br/~laffernandes/projects/kht')">Project Page</a>
%
%   The complete description of the implemented techinique can be found at the
%   above paper. If you use this implementation, please reference the above paper.
%
%   LINES = KHT(BINARY_IMAGE,'PropertyName1',PropertyValue1,'PropertyName2',PropertyValue2,...)
%   performs the KHT procedure over a single-channel 8-bit binary image and returns a
%   Nx2 array with the [rho theta] parameters (rho in pixels and theta in degrees) of
%   the N detected lines. The lines are in the form
%
%       rho = x * cosd(theta) + y * sind(theta)
%
%   and we assume that the origin of the image coordinate system is at the center
%   of the image, with the x-axis growing to the right and the y-axis growing down.
%   The resulting lines are sorted in descending order of relevance.
%
%	The expected input properties are:
%
%        'cluster_min_size' : Minimum number of pixels in the clusters of approximately
%                             collinear feature pixels. The default value is 10.
%
%   'cluster_min_deviation' : Minimum accepted distance between a feature pixel and the
%                             line segment defined by the end points of its cluster.
%                             The default value is 2.
%
%                   'delta' : Discretization step for the parameter space. The default
%                             value is 0.5.
%
%       'kernel_min_height' : Minimum height for a kernel pass the culling operation.
%                             This property is restricted to the [0,1] range. The
%                             default value is 0.002.
%
%                'n_sigmas' : Number of standard deviations used by the Gaussian kernel
%                             The default value is 2.
%
% Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
function lines = kht(binary_image,varargin)