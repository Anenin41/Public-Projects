% Author:   Konstantine Garas
% e-mail:   kgaras041@gmail.com
% Created:  Thu 10 Oct 2024 @ 21:52:53 +0200
% Modified: Tue 04 Mar 2025 @ 19:32:22 +0100

% Parameters 
L = 1;			% Length of the domain
N = 50;			% Number of grid points
k = 0.01;		% Diffusion coefficient
U = 1;			% Velocity of the fluid
phi0 = 0;		% Left boundary condition
phiL = 1;		% Right boundary condition
r = 0.95;		% Non uniform grid spacing rate (geometric spacing)
dt = 0.001;		% Time step
T = 0.5;		% Final time (seconds)
N_time = floor(T/dt);	% Number of time steps
t_method = 'explicit'	% Choose between explicit and implicit time 
			% discretization
x_method = 'central'	% Choose between Central, Upwind or Central Method A
			% discretization schemes. (central, upwind, method_A)

% Initialize grids
x_uni = linspace(0, L, N);				% Uniform grid setting
h_uni = L / (N-1);					% Uniform grid step

[x_non_uni, h_non_uni] = create_grid(0, L, N, r);	% Non-uniform grid
							% generation
% Initialize phi array in memory
% Choose between different initial conditions
% Default conditions
phi = zeros(N, 1);
phi(1) = phi0; phi(end) = phiL;				% boundary conditions
% Sine conditions (uncomment to use them)
% phi = 2 + sin((pi/L)*x_uni);
% phi(1) = 2; phi(end) = 2;				% boundary conditions

% Perform time loop and choose correct combination of discretization schemes
% based on user input
for time = 1:N_time
	if strcmp(t_method, 'explicit')
		if strcmp(x_method, 'method_A')
			phi = explicit_non_uniform(phi, h_non_uni, dt, U, k, N);
		else
			phi = explicit_uniform(phi, h_uni, dt, U, k, N, x_method);
		end
	elseif strcmp(t_method, 'implicit')
		if strcmp(x_method, 'method_A')
			phi = implicit_non_uniform(phi, h_non_uni, dt, U, k, N);
		else
			phi = implicit_uniform(phi, h_uni, dt, U, k, N, x_method);
		end
	end
	
	% Plot for evey 100 time steps
	if mod(time, 50) == 0
		if strcmp(x_method, 'method_A')
			plot(x_non_uni, phi, 'DisplayName', sprintf('t=%.2f', time*dt));
			hold on;
		else
			plot(x_uni, phi, 'DisplayName', sprintf('t=%.2f', time*dt));
			hold on;
		end
	end
end

% Final plot settings
xlabel('x');
ylabel('\phi');
title('Unsteady Convection-Diffusion Equation');
legend show;
grid on;
