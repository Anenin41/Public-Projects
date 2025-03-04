% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Mon 14 Oct 2024 @ 17:05:02 +0200
% Modified: Tue 04 Mar 2025 @ 19:32:32 +0100

% This code implements the conditions of exercise 3 in one standalone
% package, so it can be more manageable.

% Parameters
L = 1;				% Length of the computational domain
N = 100;			% Number of spatial grid points
h = L /  (N-1);			% Spatial grid step
U = 1;				% Velocity of the fluid
eta = 0.5;			% Eta
dt = (eta * h)/U;		% Time step
T = 3;				% Final time (in seconds)
N_time = ceil(T/dt);		% Number of time steps

% Create uniform grid
x = linspace(0, L, N);

% Initial condition for space
phi = 2 + sin(6 * pi * x);

% Boundary conditions for x = 0 and x = L, as functions of time
phi0 = @(t) 2 + sin(6 * pi * (0 - U * t));
phi1 = @(t) 2 + sin(6 * pi * (1 - U * t));

% Time loop
for t = 1:N_time
	% Update boundary conditions (functions of time)
	phi(1) = phi0(t*dt);
	phi(end) = phi1(t*dt);
	
	% Explicit Time & Upwind Scheme
	for i = 2:(N-1)
		phi(i) = phi(i) - eta * (phi(i) - phi(i-1));
	end

	% Plot specific time steps in the iteration
	if mod(t, 50) == 0
		plot(x, phi, 'DisplayName', sprintf('t=%.2f', t*dt));
		hold on;
	end
end

xlabel('x')
ylabel('\phi')
title('Wave Equation with Explicit & Upwind Scheme');
legend show;
grid on;
