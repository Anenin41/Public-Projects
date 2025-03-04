% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Sat 12 Oct 2024 @ 13:54:14 +0200
% Modified: Tue 04 Mar 2025 @ 19:31:21 +0100

function phi_new = explicit_non_uniform(phi, h, dt, U, k, N)
	% This function simulates the Explicit & Cetral Method A discretization
	% for the unsteady convection diffusion equation. It only works for
	% non uniform grid.
	%
	% Parameters:
	% phi: phi_old or initial guess for phi(array)
	% h: non-uniform step (array)
	% dt: time step (float)
	% U: velocity of the fluid (float)
	% k: diffusion coefficient (float)
	% N: number of grid points of the non-uniform grid (float)
	%
	% Output:
	% phi_new: phi on the new time step

	% Initial guess for phi_new & array initialization in memory
	phi_new = phi;
	eta = zeros(N-1, 1);
	delta = zeros(N-1,1);
	
	% Iteration loop
	for i = 2:(N-1)
		% Central Method A Scheme
		s = (h(i) + h(i-1));			% sum of two consecutive h elements
		p = (h(i) * h(i-1));			% product of two consecutive h elements
		% Split the terms of the equations into Left, Middle and Right
		% and then add them all together to gen the phi at the new
		% time step.
		L = ( (U*dt)/s + (2*k*dt)/(h(i-1) * s) )*phi(i-1);
		D = (1 - (2*k*dt)/p) * phi(i);
		R = ( (2*k*dt)/(h(i) * s) - (U*dt)/s )*phi(i+1);
		% Combine the above into one equation to get the desired
		% result
		phi_new(i) = L + D + R;
	end
end
