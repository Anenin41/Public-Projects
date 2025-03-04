% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Fri 11 Oct 2024 @ 19:47:29 +0200
% Modified: Tue 04 Mar 2025 @ 19:31:36 +0100

function phi_new = explicit_uniform(phi, h, dt, U, k, N, method)
	% This function simulates the explicit solvers for the unsteady
	% convection diffusion equation. It only works for uniform grid
	% and Central / Upwind discretization schemes.
	%
	% Parameters:
	% phi: phi_old or initial guess for phi (array)
	% h: uniform step (float)
	% dt: time step (float)
	% U: velocity of the fluid (float)
	% k: diffusion coefficient (float)
	% N: number of grid points (float)
	% method: 'central' or 'upwind', user prompt on which method to use
	%
	% Output:
	% phi_new: phi on the new time step
	
	% Initial guess for phi_new & initialization in memory
	phi_new = phi;
	
	% Iterations for Central & Upwind
	if strcmp(method, 'central')
		% Central Scheme
		eta = (U*dt)/h;				% Define eta
		delta = (2*k*dt)/(h^2);			% Define delta
		for i = 2:(N-1)
		% Split the operations in 3 parts, and then add them all
		% together.
			L = (eta/2 + delta/2)*phi(i-1);	% Left term
			D = (1-delta)*phi(i);		% Middle term
			R = (-eta/2 + delta/2)*phi(i+1);% Right term
			phi_new(i) = L+D+R;
		end
	elseif strcmp(method, 'upwind')
		% Upwind Scheme
		eta = (U*dt)/h;				% Define eta
		delta = (2*k*dt)/(h^2);			% Define delta
		for i = 2:(N-1)
		% Split the operations in 3 parts, and then add them all
		% together
			L = (eta + delta/2)*phi(i-1);	% Left term
			D = (1-delta-eta)*phi(i);	% Middle Term
			R = (delta/2)*phi(i+1);		% Right Term
			phi_new(i) = L+D+R;
		end
	end
end
