% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Sat 12 Oct 2024 @ 14:04:33 +0200
% Modified: Tue 04 Mar 2025 @ 19:31:51 +0100

function phi_new = implicit_non_uniform(phi, h, dt, U, k, N)
	% This function simulates the Implicit Time & Central Method A 
	% discretization for non-uniform grids.
	%
	% Parameters:
	% phi: phi_old or initial guess for phi (array)
	% h: non-uniform step (array)
	% dt: time step (float)
	% U: velocity of the fluid (float)
	% k: diffusion coefficient (float)
	% N: number of grid points (float)
	%
	% Output:
	% phi_new: phi on the new time step
	
	% Initialize arrays in memory (will use TDMA)
	L = zeros(N-1, 1);				% Lower diagonal (N-1)
	D = zeros(N, 1);				% Main diagonal (N)
	R = zeros(N-1, 1);				% Upper diagonal (N-1)
	eta = zeros(N-1,1);
	delta = zeros(N-1,1);
	
	% Iteration loop
	for i = 2:(N-1)
	% Central Method A scheme
		s = h(i-1) + h(i);		% sum of two consecutive 
						% elements of the h array
		p = h(i-1) * h(i);		% product of two consecutive
						% elements of the h array
		% Calculate the coefficient matrix (tridiagonal structure)
		L(i) = ( -(U*dt)/s - (2*k*dt)/(h(i-1) * s) );	% Lower diagonal
		D(i) = (1 + (2*k*dt)/p);			% Main diagonal
		R(i) = ( (U*dt)/s - (2*k*dt)/(h(i) * s) );	% Upper diagonal
	end
	
	% Set boundary conditions
	D(1) = 1; D(end) = 1;
	
	% Solve the linear system
	phi_new = tdma(L,D,R,phi);
end	
