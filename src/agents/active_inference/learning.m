clear 
clc
rng(1992)

%49834)%4981)
addpath ~\bayes
addpath ~\spm\
addpath ~\toolbox\DEM

%Likelihood learning: 

mdp = likelihood_model();   
[MDP(1:10)]    = deal(mdp);    
MDP  = spm_MDP_VB_X(MDP);

save('~\learning_A.mat', 'MDP');

% Belief states for trajectories:
one = transpose(reshape(sum(MDP(1).X{1},2),[3,3]));
two = transpose(reshape(sum(MDP(2).X{1},2),[3,3]));
three = transpose(reshape(sum(MDP(3).X{1},2),[3,3]));
four = transpose(reshape(sum(MDP(4).X{1},2),[3,3]));
colormap('default');
imagesc(one);


% Preference learning
clear mdp MDP
mdp = preference_model();   
[MDP(1:10)]    = deal(mdp);    
MDP  = spm_MDP_VB_X(MDP);

save('~\learning_C.mat', 'MDP');

colormap('default');
imagesc(MDP(10).c{1,2}(:,1:4))


% Preference learning
rng(8)
clear mdp MDP
mdp = preference_model();   
[MDP(1:10)]    = deal(mdp);    
MDP  = spm_MDP_VB_X(MDP);
save('~\learning_C_pos.mat', 'MDP');

colormap('default');
imagesc(MDP(10).c{1,2}(:,1:4))


% Preference and likelihood learning:
c = mdp.c;
clear mdp MDP
mdp = likelihood_model();  
mdp.c =c; 
[MDP(1:20)]    = deal(mdp);    
MDP  = spm_MDP_VB_X(MDP);

