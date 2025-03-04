% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com // k.gkaras@student.rug.nl
% Created:  Fri 11 Oct 2024 @ 20:08:33 +0200
% Modified: Thu 14 Nov 2024 @ 21:09:36 +0100

function phi_new = implicit_uniform(phi, h, dt, U, k, N, method)
	% This function simulates the implicit solvers for the unsteady 
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
	
	% Initialize arrays in memory (will use TDMA)
	L = zeros(N-1, 1);				% Lower diagonal (N-1)
	D = zeros(N, 1);				% Main diagonal (N)
	R = zeros(N-1,1);				% Upper Diagonal (N-1)
	
	% Iteration loop
	if strcmp(method, 'central')
	% Central Scheme
		eta = (U*dt)/h;				% Define eta
		delta = (2*k*dt)/(h^2);			% Define delta
		for i = 2:(N-1)
			L(i) = (-eta/2 - delta/2);	% Lower diagonal
			D(i) = (delta + 1);		% Main diagonal
			R(i) = (eta/2 - delta/2);	% Upper diagonal
		end
	elseif strcmp(method, 'upwind')
	% Upwind Scheme
		eta = (U*dt)/h;				% Define eta
		delta = (2*k*dt)/(h^2);			% Define delta
		for i = 2:(N-1)
			L(i) = (-eta - delta/2);	% Lower diagonal
			D(i) = (delta + eta + 1);	% Main diagonal
			R(i) = (-delta/2);		% Upper diagonal
		end
	end
	
	% Set boundary conditions
	D(1) = 1 ; D(end) = 1;

	% Solve the linear system
	phi_new = tdma(L,D,R,phi);
end
