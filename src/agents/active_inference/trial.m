
clear 
clc

addpath D:\PhD\Code\bayes
addpath D:\PhD\Code\spm\
addpath D:\PhD\Code\spm\toolbox\DEM

num_trials = 200;
num_episodes = 500;
i   = [1:20,120:140,250:450];        % change context in a couple of trials
    
%storing results:
trwp = zeros(num_episodes, num_trials);
trwop = zeros(num_episodes, num_trials);

% with preferences: 
for j = 1:num_trials
    
    clear mdp MDP
    mdp = model();
    [MDP(1:num_episodes)]    = deal(mdp);      % create structure array
    [MDP(i).s]     = deal([1 1]');          % deal context changes for true state
    MDP  = spm_MDP_VB_X(MDP);
    for i = 1:500
        trwp(i,j) = MDP(i).o(2,4);
    end    
end

% without preferences: 
for j = 1:num_trials
    
    clear mdp MDP
    mdp = model();
    mdp.C{1}(1)= 0;
    mdp.C{2}= [0 0 0]';
    [MDP(1:num_episodes)]    = deal(mdp);      % create structure array
    [MDP(i).s]     = deal([1 1]');          % deal context changes for true state
    MDP  = spm_MDP_VB_X(MDP);
    
    for i = 1:500
        trwop(i,j) = MDP(i).o(2,4);
    end    
end


%plotting in python
csvwrite('D:\PhD\Code\bayes\trwp.csv',trwp)
csvwrite('D:\PhD\Code\bayes\trwop.csv',trwop)


    
%storing results:
trwp_det = zeros(num_episodes, num_trials);
trwop_det = zeros(num_episodes, num_trials);

% with preferences: 
for j = 1:num_trials
    
    clear mdp_det MDP_det
    disp(i)
    mdp_det = model();
    [MDP_det(1:num_episodes)]    = deal(mdp_det);      % create structure array
    MDP_det  = spm_MDP_VB_X(MDP_det);
    for i = 1:500
        trwp_det(i,j) = MDP_det(i).o(2,4);
    end    
end

% without preferences: 
for j = 1:num_trials
    
    disp(i)
    clear mdp_det MDP_det
    mdp_det = model();
    mdp_det.C{1}(1)= 0;
    mdp_det.C{2}= [0 0 0]';
    [MDP_det(1:num_episodes)]    = deal(mdp_det);      % create structure array
    MDP_det  = spm_MDP_VB_X(MDP_det);
    
    for i = 1:500
        trwop_det(i,j) = MDP_det(i).o(2,4);
    end    
end
