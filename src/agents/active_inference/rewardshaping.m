clear 
clc
rng(4981)

%%% Reward shaping:

num_trials = 1;
num_episodes = 100;

% Negative reward for living
trwop_priors_score_ll = zeros(num_episodes, num_trials);

mdp = {};
 for j = 1:num_trials
      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model();   
             MDP.C{1}= [-5 -5 -5 -5 -5 0 -5 0 -5]';
             MDP.C{2}= [+5 0 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_ll(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_ll_modified2.csv',trwop_priors_score_ll)
clear mdp

% Negative reward for living & hit:
trwop_priors_score_llh = zeros(num_episodes, num_trials);
      mdp = {};
 for j = 1:num_trials
      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model();   
             MDP.C{1}= [-5 -5 -5 -5 -5 0 -5 0 -5]';
             MDP.C{2}= [+4 -4 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_llh(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_llh_modified2.csv',trwop_priors_score_llh)
clear mdp


% Negative reward for hit:
trwop_priors_score_h = zeros(num_episodes, num_trials);
      mdp = {};
 for j = 1:num_trials

      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model();   
             MDP.C{1}(1) = 0;
             MDP.C{2}= [+5 -5 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_h(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_h_modified2.csv',trwop_priors_score_h)


% Negative reward for (just) hit:
trwop_priors_score_ph = zeros(num_episodes, num_trials);
      mdp = {};
 for j = 1:num_trials

      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model_rs();   
             MDP.C{1}(1) = 0;
             MDP.C{2}= [0 -5 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_ph(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_ph_modified2.csv',trwop_priors_score_ph)


% No Reward:
trwop_priors_score_pll = zeros(num_episodes, num_trials);
      mdp = {};
 for j = 1:num_trials

      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model();
             MDP.C{1}(1) = 0;
             MDP.C{2}= [0 0 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_pll(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_no_reward_modified2.csv',trwop_priors_score_pll)



% Baseline:
trwop_priors_score_pll = zeros(num_episodes, num_trials);
mdp = {};

 for j = 1:num_trials

      fprintf('%i\n', 100*(j/num_trials))    
      for i = 1:num_episodes
             
             MDP = model();
             MDP.C{1}(1) = 0;
             MDP.C{2}= [+5 0 0]';
             
             if i > 1
                % using the posterior from previous trial as the 
                % prior for this trial
                MDP.D{2} = X; 
             end
             
             MDP  = spm_MDP_VB_X(MDP);     
             mdp{i,j} = MDP;
             trwop_priors_score_pll(i,j) = MDP.o(2,MDP.T);        
             
             % keeping the posterior
             X = MDP.X{2}(:,end);
            
     end    
 end

csvwrite('~\trwop_priors_score_pll_modified2.csv',trwop_priors_score_pll)

