% Copyright (C) Leandro A. F. Fernandes and Manuel M. Oliveira
%
% author     : Fernandes, Leandro A. F.
% e-mail     : laffernandes@ic.uff.br
% home page  : http://www.ic.uff.br/~laffernandes
% 
% This file is part of the reference implementation of the Kernel-Based
% Hough Transform (KHT). The complete description of the implemented
% techinique can be found at:
% 
%     Leandro A. F. Fernandes, Manuel M. Oliveira
%     Real-time line detection through an improved Hough transform
%     voting scheme, Pattern Recognition (PR), Elsevier, 41:1, 2008,
%     pp. 299-314.
% 
%     DOI.........: https://doi.org/10.1016/j.patcog.2007.04.003
%     Project Page: http://www.ic.uff.br/~laffernandes/projects/kht
%     Repository..: https://github.com/laffernandes/kht
% 
% KHT is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% KHT is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
% License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with KHT. If not, see <https://www.gnu.org/licenses/>.

close all
clear
clc

% Set sample image files and number of most relevant lines.
[base_folder,~,~] = fileparts(mfilename('fullpath'));
filenames = {'simple.jpg','chess.jpg','road.jpg','wall.jpg','board.jpg','church.jpg','building.jpg','beach.jpg'};
relevant_lines = [8,25,15,36,38,40,19,19];

% Process each one of the images.
for i=1:length(filenames)
    % Load input image.
    im = imread(fullfile(base_folder,'..','..','extra',filenames{i}));
    [height,width,~] = size(im);

    % Convert the input image to a binary edge image.
    bw = uint8(edge(rgb2gray(im),'canny'));

    % Call the kernel-base Hough transform function.
    lines = kht(bw);
    
    % Show current image and its most relevant detected lines.
    figure('Name',sprintf('KHT - Image ''%s'' - %d most relevant lines',filenames{i},relevant_lines(i)))
    warning('off','Images:initSize:adjustingMag')
    imshow(im)
    warning('on','Images:initSize:adjustingMag')

    hold on
    for j=1:relevant_lines(i)
        % Convert from KHT to MATLAB's coordinate system conventions.
        % The KHT implementation assumes row-major memory alignment for
        % images. Also, it assumes that the origin of the image coordinate
        % system is at the center of the image, with the x-axis growing to
        % the right and the y-axis growing down.
        if sind(lines(j,2)) ~= 0
            y = [-height/2,height/2-1];
            x = (lines(j,1)-y*cosd(lines(j,2)))/sind(lines(j,2));
        else
            y = [lines(j,1),lines(j,1)];
            x = [-width/2,width/2-1];
        end
        x = x+width/2+1;
        y = y+height/2+1;
        patch(x,y,[1,1,0],'EdgeColor','y','LineWidth',0.5);
    end
    hold off
end