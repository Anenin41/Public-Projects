% This function creates a non uniform grid with rate of change r %
% Author: Konstantinos Garas
% Email: kgaras041@gmail.com
% Last Modified: Tue 04 Mar 2025 @ 19:30:46 +0100

function [x,h] = create_grid(x0, xL, N, r)
    % Create a non-uniform grid where the spacing between points shrinks by 
    % a factor r.
    %
    % Inputs:
    % x0:   Start of the domain (first grid point)
    % xL:   End of the domain (last grid point)
    % N:    Number of grid points 
    % r:    Rate of decrease between consecutive grid points
    %
    % Output:
    % x:    Non-uniform grid (vector of size N)
    % h:    Step between each grid point (vector of size N-1)
    
    % Initialize the grid vectors in memory
    x = zeros(1, N);
    h = zeros(1,N-1);
    x(1) = x0;   % First grid point
    
    % Calculate each subsequent point based on the rate r
    for i = 2:N
        x(i) = x(i-1) + (xL - x0) * r^(i-2);
    end
    
    % Normalize the grid to ensure it fits in [x0, xL]
    x = x0 + (x - x(1)) * (xL - x0) / (x(end) - x(1));
    h = diff(x);       % Use the vpa() package which ensures higher precision 
		       %in calculations
end
