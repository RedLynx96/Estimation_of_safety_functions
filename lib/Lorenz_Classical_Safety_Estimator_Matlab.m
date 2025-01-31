clc
clear all

%% This file will help you visualize the Lorenz system under the application of partial control
%% It will also output a file with the variables 'xVectors' (sampled data) and 'Us'
%% (classically computed safety function) predict and compare with the transformer-based model behaviour

%Read line 254 in case you want to run predictions on a controlled orbit
%from a custom safety function such as the one predicted by the
%transformer-based model.

%Lorenz Parameters
sigma = 10;
r = 20;
b = 8/3;

% Noise intensity during the integration of the orbits
noise_intensity = 0.1; 

% Integration parameters
tspan = [0 50];
h = 0.001; % Integration step
n = (tspan(2)-tspan(1))/h;
t = linspace(tspan(1),tspan(2),n);

% Generate n_trayectories random initial conditions 
n_trayectorias = 2000;
x0_range = [-10 10];
rng(1);
x0_array = x0_range(1) + (x0_range(2)-x0_range(1))*rand(3,n_trayectorias);

x_all = zeros(3,n,n_trayectorias);

% Trajectory integration
parfor traj = 1:n_trayectorias
    x = zeros(3,n);
    x(:,1) = x0_array(:,traj);
    
    % Runge-Kutta 4
    for i = 1:n-1
        k1 = lorenz(t(i), x(:,i), sigma, r, b);
        k2 = lorenz(t(i)+h/2, x(:,i)+h*k1/2, sigma, r, b);
        k3 = lorenz(t(i)+h/2, x(:,i)+h*k2/2, sigma, r, b);
        k4 = lorenz(t(i)+h, x(:,i)+h*k3, sigma, r, b);
        
        % Deterministic term
        deterministic = h*(k1 + 2*k2 + 2*k3 + k4)/6;
        
        % Stochastic term
        stochastic = noise_intensity * sqrt(h) * randn(3,1);
        
        x(:,i+1) = x(:,i) + deterministic + stochastic;
    end
    
    x_all(:,:,traj) = x;
end

% First figure
fig1 = figure(1);
set(fig1, 'Position', [100, 100, 1000, 400]);

% Encontrar puntos que cumplen las condiciones antes de los subplots
all_points = [];
all_y_abs = [];
all_y_next_abs = [];

for traj = 1:n_trayectorias
    y = x_all(2,:,traj);
    z = x_all(3,:,traj);
    
    % Poincaré section
    indices = find(z > 20 & abs(y) > 12 & abs(y) < 22);
    
    diff_indices = diff(indices);
    gap_indices = find(diff_indices > 2);
    
    if ~isempty(gap_indices)
        valid_indices = [indices(1), indices(gap_indices+1)];
        valid_indices = valid_indices(2:end);
        
        if length(valid_indices) > 1
            all_points = [all_points, squeeze(x_all(:,valid_indices,traj))];
            
            y_values = abs(y(valid_indices(1:end-1)));
            y_next_values = abs(y(valid_indices(2:end)));
            
            if all(y_next_values < 23)
                all_y_abs = [all_y_abs, y_values];
                all_y_next_abs = [all_y_next_abs, y_next_values];
            end
        end
    end
end

% Subplot 1: Attractor
subplot(1,2,1);
hold on;
colors = jet(n_trayectorias);
for traj = 1:n_trayectorias
    plot3(x_all(1,:,traj), x_all(2,:,traj), x_all(3,:,traj), 'Color', colors(traj,:));
end

num_points = size(all_points, 2);
num_points_to_show = round(num_points);
random_indices = randperm(num_points, num_points_to_show);

plot3(all_points(1,random_indices), all_points(2,random_indices), all_points(3,random_indices), ...
    'k.', 'MarkerSize', 30);

grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Lorenz attractor with Poincaré section');
view(90, 0);
hold off

subplot(1,2,2);
hold on;
plot(all_y_abs, all_y_next_abs, 'k.', 'MarkerSize', 2);

x_diag = 12:0.1:19;
plot(x_diag, x_diag, '--k', 'LineWidth', 1);

%Split points in two branches

pto_medio=15.754;
left_branch = all_y_abs < pto_medio;
right_branch = all_y_abs >= pto_medio;

left_x = all_y_abs(left_branch);
left_y = all_y_next_abs(left_branch);
[left_x, sort_idx] = sort(left_x);
left_y = left_y(sort_idx);

right_x = all_y_abs(right_branch);
right_y = all_y_next_abs(right_branch);
[right_x, sort_idx] = sort(right_x);
right_y = right_y(sort_idx);

left_x = left_x(:);
left_y = left_y(:);
right_x = right_x(:);
right_y = right_y(:);

% Cloud fit to obtain y = f(x) and noise ranges, we split the cloud on two
% branches, left and right

left_branch = all_y_abs < pto_medio;
right_branch = all_y_abs >= pto_medio;

left_x = all_y_abs(left_branch);
left_y = all_y_next_abs(left_branch);
[left_x, sort_idx] = sort(left_x);
left_y = left_y(sort_idx);

right_x = all_y_abs(right_branch);
right_y = all_y_next_abs(right_branch);
[right_x, sort_idx] = sort(right_x);
right_y = right_y(sort_idx);

% Random forest settings to fit the curves
min_leaf_size = 1;
max_num_splits = 100;
n_trees = 300;

Tbl_left = table(left_x(:), left_y(:), 'VariableNames', {'x', 'y'});
Tbl_right = table(right_x(:), right_y(:), 'VariableNames', {'x', 'y'});

x_plot_left = linspace(min(left_x), max(left_x), 100)';
x_plot_right = linspace(min(right_x), max(right_x), 100)';

X_eval_left = table(x_plot_left, 'VariableNames', {'x'});
X_eval_right = table(x_plot_right, 'VariableNames', {'x'});

rng('default');
Mdl_left = TreeBagger(n_trees, Tbl_left, 'y', 'Method', 'regression', ...
    'MinLeafSize', min_leaf_size, ...
    'MaxNumSplits', max_num_splits);

Mdl_right = TreeBagger(n_trees, Tbl_right, 'y', 'Method', 'regression', ...
    'MinLeafSize', min_leaf_size, ...
    'MaxNumSplits', max_num_splits);

y_median_left = predict(Mdl_left, X_eval_left);
y_median_right = predict(Mdl_right, X_eval_right);

quant_left = quantilePredict(Mdl_left, X_eval_left, 'Quantile', [0.02 0.98]);
quant_right = quantilePredict(Mdl_right, X_eval_right, 'Quantile', [0.02 0.98]);

%span = 0.4; % Adjust between 0 and 1 to adjust the smoothness
span=noise_intensity*3.33+0.067; %heuristic rel
y_upper_left = smooth(x_plot_left, quant_left(:,2), span, 'loess');
y_lower_left = smooth(x_plot_left, quant_left(:,1), span, 'loess');
y_median_left = smooth(x_plot_left, y_median_left, span, 'loess');

y_upper_right = smooth(x_plot_right, quant_right(:,2), span, 'loess');
y_lower_right = smooth(x_plot_right, quant_right(:,1), span, 'loess');
y_median_right = smooth(x_plot_right, y_median_right, span, 'loess');

% Plot the curves
plot(x_plot_left, y_upper_left, 'r-', 'LineWidth', 1);
plot(x_plot_left, y_lower_left, 'r-', 'LineWidth', 1);
plot(x_plot_right, y_upper_right, 'r-', 'LineWidth', 1);
plot(x_plot_right, y_lower_right, 'r-', 'LineWidth', 1);
plot(x_plot_left, y_median_left, 'b-', 'LineWidth', 1);
plot(x_plot_right, y_median_right, 'b-', 'LineWidth', 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find diagonal intersection
x_test = 13.8:0.01:14.2; %adapt to any case
y_left_interp = interp1(left_x, left_y, x_test, 'linear');
[~, idx] = min(abs(x_test - y_left_interp));
x0 = x_test(idx);
y0 = y_left_interp(idx);

y_right_interp = interp1(right_y, right_x, y0, 'linear');
x1 = y_right_interp;
y1 = y0;

square_size = x1 - x0;

square_x = [x0, x1, x1, x0, x0];
square_y = [y0, y1, y1 + square_size, y0 + square_size, y0];

plot(square_x, square_y, 'k-', 'LineWidth', 1);

xlabel('$|y_n|$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$|y_{n+1}|$', 'Interpreter', 'latex', 'FontSize', 15);
title(['Continuous noise added during integraton $I= ' num2str(noise_intensity) '$'], ...
    'Interpreter', 'latex', 'FontSize', 13);
grid on;
axis equal;

hold off

%%% Estimation xVectors to obtained sampled data
%%% xVectors is scaled down to [0,1], remember to later de-normalize it!

xVectors = [all_y_abs; all_y_next_abs]';

min_x = min(square_x);
max_x = max(square_x);
min_y = min(square_y);
max_y = max(square_y);

is_within_x = ((xVectors(:,1) >= min_x) & (xVectors(:,1) <= max_x)) | ((xVectors(:,1) == -1) & (xVectors(:,2) == -1));
xVectors = xVectors(is_within_x, :);

% Step 2: add [-1,-1] at the end of each orbit
newxVectors = xVectors(1, :);
% Iterate through each row of xVectors
for i = 1:size(xVectors, 1)-1
    current_end = xVectors(i, 2);       % x_{n+1} of the current row
    next_start = xVectors(i+1, 1);     % x_n of the next row
    
    if current_end ~= next_start
        % If there's a discontinuity, append [-1, -1]
        newxVectors = [newxVectors; -1, -1];
    end
    
    % Append the next row
    newxVectors = [newxVectors; xVectors(i+1, :)];
end

xVectors = newxVectors;

% Step 3: Scale the x and y values
% For x: Scale to [0, 1]
scaled_x = (xVectors(:,1) - min_x) / (max_x - min_x);

% For y: Scale to [0, 1], allowing values outside to be <0 or >1
scaled_y = (xVectors(:,2) - min_y) / (max_y - min_y);

% Identify rows where the pair is [-1, -1]
exclude_idx = (xVectors(:,1) == -1) & (xVectors(:,2) == -1);

% Retain [-1, -1] in the scaled arrays by setting them back to -1
scaled_x(exclude_idx) = -1;
scaled_y(exclude_idx) = -1;

% Combine the scaled x and y into a new variable
xVectors = [scaled_x, scaled_y];
length(xVectors)

%%%%SAFETY FUNCTION COMPUTATION WITH VARIABLE NOISE

n_points = 1000;
x_sorted = linspace(x0, x1, n_points)';

x_left_mask = x_sorted < pto_medio;
x_right_mask = x_sorted >= pto_medio;

y_upper_final = zeros(size(x_sorted));
y_lower_final = zeros(size(x_sorted));

y_upper_final(x_left_mask) = interp1(x_plot_left, y_upper_left, x_sorted(x_left_mask), 'linear', 'extrap');
y_lower_final(x_left_mask) = interp1(x_plot_left, y_lower_left, x_sorted(x_left_mask), 'linear', 'extrap');

y_upper_final(x_right_mask) = interp1(x_plot_right, y_upper_right, x_sorted(x_right_mask), 'linear', 'extrap');
y_lower_final(x_right_mask) = interp1(x_plot_right, y_lower_right, x_sorted(x_right_mask), 'linear', 'extrap');

for i = 1:length(x_sorted)
    y_upper_final(i) = max(y_upper_final(i), y_lower_final(i));
    y_lower_final(i) = min(y_upper_final(i), y_lower_final(i));
end

cglobal1 = curvamaxdirecto(x_sorted, y_upper_final, y_lower_final, x0, x1);%

%LOAD HERE AN ARRAY IN CASE YOU WANT TO THE ORBIT BEHAVIOUR ON A CUSTOM
%SAFETY FUNCTION SUCH AS THE ONE PREDICTED BY THE TRANSFORMER-BASED MODEL

%cglobal1 = load('Sample_noise_01_ML.mat', 'cglobal1');
%cglobal1 = (struct2array(cglobal1));% * (max(square_x) - min(square_x)))';

figure(3)
clf;
hold on
plot(x_sorted, cglobal1)
xlabel('x')
ylabel('Safety Function')
grid on
hold off

%%%%%%%%%%%%%%%%%%%%%%%TRAJECTORY CONTROL%%%%%%%%%%%%%%%%%%%%

rama1 = [];
rama2 = [];

for i = 1:size(all_points, 2)
    if all_points(3,i) > 20 
        if all_points(2,i) < 0  
            rama1 = [rama1; all_points(1,i) all_points(2,i)];
        else 
            rama2 = [rama2; all_points(1,i) all_points(2,i)];
        end
    end
end

[~, idx] = sort(rama1(:,2));
rama1 = rama1(idx,:);
[~, idx] = sort(rama2(:,2));
rama2 = rama2(idx,:);

% polynomial fit for eeach branch
orden_poly = 8;
p1 = polyfit(rama1(:,2), rama1(:,1), orden_poly); % x = f(y) para rama1
p2 = polyfit(rama2(:,2), rama2(:,1), orden_poly); % x = f(y) para rama2

%%%%%%%%%%%%%%%%%%%%%%%%%%% Trajectory and control

Vind = 1:length(x_sorted);

% Initial condition
x0 = [-11.54; -14.68; 20];
current_point = x0;

max_points = 1e6;  
x_controlled = zeros(3, max_points);
x_controlled(:,1) = x0;
controls_used = zeros(1000, 1);  % for the 1000 control values we want
ynext_values = zeros(1000, 1);  
num_controls = 0;
point_counter = 1;

% Variable to detect the crossover with z=20
prev_point = current_point;

while num_controls < 1000 %Number of controlled steps
num_controls
    % Calcular siguiente paso con Runge-Kutta 4
    k1 = lorenz(0, current_point, sigma, r, b);
    k2 = lorenz(0, current_point+h*k1/2, sigma, r, b);
    k3 = lorenz(0, current_point+h*k2/2, sigma, r, b);
    k4 = lorenz(0, current_point+h*k3, sigma, r, b);
    
    deterministic = h*(k1 + 2*k2 + 2*k3 + k4)/6;
    stochastic = noise_intensity * sqrt(h) * randn(3,1);
    
    next_point = current_point + deterministic + stochastic;
    
    point_counter = point_counter + 1;
    if point_counter <= max_points
        x_controlled(:,point_counter) = current_point;
    end
    
    if (prev_point(3) < 20 && next_point(3) >= 20)
        if abs(next_point(2)) > 12 && abs(next_point(2)) < 22
            num_controls = num_controls + 1;
            
            en_rama1 = next_point(2) < 0;
            yabs = abs(next_point(2));
            
            [ynext, control] = siguiente_iteracion(yabs, x_sorted, cglobal1, Vind);
            
            ynext_values(num_controls) = ynext;
            
            if en_rama1
                ynext = -ynext;
            end
            
            controls_used(num_controls) = control;
            
            if ynext < 0
                xnext = polyval(p1, ynext);
            else
                xnext = polyval(p2, ynext);
            end
            
            current_point = [xnext; ynext; next_point(3)];
            prev_point = current_point;
            continue;
        end
    end
    
    prev_point = current_point;
    current_point = next_point;
end

controls_used = controls_used(1:num_controls);
ynext_values = ynext_values(1:num_controls);

figure(7);
x_controlled(:, all(x_controlled == 0, 1)) = [];
plot3(x_controlled(1,:), x_controlled(2,:), x_controlled(3,:),'r','linewidt',0.5);
grid on;
xlabel('x');
ylabel('y');
zlabel('z');
title('Controlled Lorenz Trajectory');

figure(8);
hold on;
b = bar(1:length(controls_used), abs(controls_used));
set(b, 'FaceColor', 'b', 'BarWidth', 1);
min_cglobal1 = min(cglobal1);
plot([1 length(controls_used)], [min_cglobal1 min_cglobal1], 'k-', 'LineWidth', 1);

text(length(controls_used)/2, min_cglobal1-0.007, '$\min(\mathrm{safety~function})$', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'Interpreter', 'latex', 'FontSize', 11, 'Color', 'k');

xlabel('Control Step');
ylabel('|Control|');
title('Control values applied');
grid on;
set(gca, 'XLim', [1 length(controls_used)]);
hold off;

x_plot_right(1) = x_plot_left(100);
y_median_right(1) = y_median_left(100);

fig9 = figure(9);
set(fig9, 'Position', [1247         271         465         590]);
plot(x_plot_left, y_median_left, 'b-', 'LineWidth', 1);
hold on;
plot(x_plot_right, y_median_right, 'b-', 'LineWidth', 1);

plot(square_x, square_y, 'k-', 'LineWidth', 1);

x_diag = 12:0.1:19;
plot(x_diag, x_diag, '--k', 'LineWidth', 1);

stairs(ynext_values(1:end-1),ynext_values(2:end), 'r-', 'LineWidth', 0.5);

xlabel('$|y_n|$', 'Interpreter', 'latex', 'FontSize', 15);
ylabel('$|y_{n+1}|$', 'Interpreter', 'latex', 'FontSize', 15);
title('Controlled map', 'FontSize', 13);
grid on;
axis([13.5 18 13 19]);
hold off;

save(sprintf('LorenzData_noise=%.2f.mat', noise_intensity), 'xVectors');
Us = cglobal1; 
save(sprintf('LorenzData_noise=%.2f.mat', noise_intensity), 'Us', '-append');

function [y_next, control] = siguiente_iteracion(y1, Q, U,Vind)
   
    controles = abs(y1 - Q);
    [control_max] = max(controles, U);
    [control_min] = min(control_max);
    
    sel = control_max == control_min; % Since there are several control_min repeated, we choose the smaller avainable control 
    control_sel = controles(sel);
    Vind_sel = Vind(sel);
    [control, ind] = min(control_sel);
    indfinal = Vind_sel(ind);
    
    y_next = Q(indfinal);
end

function dxdt = lorenz(t, x, sigma, r, b)
dxdt = zeros(3,1);
dxdt(1) = sigma*(x(2) - x(1));
dxdt(2) = r*x(1) - x(2) - x(1)*x(3);
dxdt(3) = x(1)*x(2) - b*x(3);
end


function cglobal1 = curvamaxdirecto(x, yup, ydown, Q1, Q2)
    % Make sure that each vector is column shaped
    x = x(:);
    yup = yup(:);
    ydown = ydown(:);
    
    Nx = length(x);
    buenos = (x >= Q1 & x <= Q2);
    
    n_points = 5000;
    yimagen = linspace(min(ydown), max(yup), n_points)';
    
    cglobal1 = zeros(Nx, 1);
    cglobal2 = zeros(Nx, 1);
    
    for rep = 1:20
        clocaly = zeros(length(yimagen), 1);
        
        for k = 1:length(yimagen)
            disty = abs(yimagen(k) - x(buenos));
            maxbothy = max(cglobal1(buenos), disty);
            [clocaly(k), ~] = min(maxbothy);
        end
        
        for k1 = 1:Nx
            indydown = max(1, find(yimagen >= ydown(k1), 1));
            indyup = min(length(yimagen), find(yimagen <= yup(k1), 1, 'last'));
            
            if isempty(indydown)
                indydown = 1;
            end
            if isempty(indyup)
                indyup = length(yimagen);
            end
            
            if indydown <= indyup
                cglobal2(k1) = max(max(clocaly(indydown:indyup)), cglobal1(k1));
            else
                cglobal2(k1) = cglobal1(k1);
            end
        end
        
        diff = norm(cglobal1 - cglobal2);
        fprintf('Iteración %d, diferencia: %f\n', rep, diff);
        
        if diff < 1e-6
            break;
        end
        
        cglobal1 = cglobal2;
    end
end
