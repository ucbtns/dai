clear 
clc

addpath ~\bayes
addpath ~\spm\
addpath ~\toolbox\DEM

num_trials = 200;
num_episodes = 500;
z  = [1:20,120:140, 250:450];        % change context in a couple of trials

% With preferences:
trwp = zeros(num_episodes, num_trials);
 for j = 1:num_trials
      mdp = {};
      
      for i = 1:num_episodes
             MDP = model();   
            
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             if sum(z==i) ==1
                [MDP.s]     = [1 2]';
             else
               [MDP.s]     = [1 1]';
             end

             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i} = MDP;
             trwp(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end
 
  csvwrite('~\trwp_modified.csv',trwp)
  
  
% Without preferences: 
  trwop = zeros(num_episodes, num_trials);
  
  for j = 1:num_trials
      mdp = {};
      
      for i = 1:num_episodes
             MDP = model();   
             MDP.C{1}(1)= 0;
             MDP.C{2}= [0 0 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             if sum(z==i) ==1
                   [MDP.s]     = [1 2]';
             else
                   [MDP.s]     = [1 1]';
             end

             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i} = MDP;
             trwop(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end


%plotting in python
csvwrite('~\trwop_modified.csv',trwop)


% Deterministic: 
%storing results:
trwp_det = zeros(num_episodes, num_trials);
trwop_det = zeros(num_episodes, num_trials);

% With preferences:
 for j = 1:num_trials
      mdp = {};
      
      for i = 1:num_episodes
             MDP = model();   
            
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i} = MDP;
             trwp_det(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end
 
 csvwrite('~\trwp_det_modified.csv',trwp_det)

% without preferences: 
 for j = 1:num_trials
      mdp = {};
      
      for i = 1:num_episodes
             MDP = model();  
             MDP.C{1}(1)= 0;
             MDP.C{2}= [0 0 0]';
            
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i} = MDP;
             trwop_det(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end
csvwrite('~\trwop_det_modfiied.csv',trwop_det)


