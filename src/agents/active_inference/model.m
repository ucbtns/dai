function mdp = model
clear;
clc

addpath D:\PhD\Code\spm12\toolbox\
addpath D:\PhD\Code\spm12

% MDP:

% P(s)
D{1} = [128 zeros(1,8)]';                % Location:1:25 % control state
D{2} = [2 2]';                               % Context: {1, 2}

% P(o|s)
Nf = numel(D); 

% Number of factor in each state
for f = 1:Nf
    Ns(f) = numel(D{f}); 
end

No    = [9 3];   % Location 1:9; 
                         % Reward [Positive, Negative, Nothing]
                
% Total outcome levels: 3
Ng    = numel(No); 

% Matrix with all possible combinations (outcomes;states)'
e = .1e-2;
for g = 1:Ng
    A{g} = ones([No(g),Ns])*e; 
end
          
env1 =  ['S' 'F' 'F' 'F' 'F' 'G' 'F' 'H' 'F'];
env2 =  ['S' 'F' 'F' 'F' 'F' 'H' 'F' 'G' 'F'];

type{'S'} = 1;
type{'F'} = 1;
type{'H'} = 1;
type{'G'} = 1;


aa      = .98;
bb      = 1 - aa;

for f1 = 1:Ns(1) 
    for f2 = 1:Ns(2)  
                                    % beliefs about position
                                    A{1}(f1,f1,:) = 1;
                                    
                                    % beliefs about environment 
                                    if f2 == 1
                                        env = env1;
                                    else
                                        env = env2;
                                    end
                                    
                                    % Adding the slippery part:
                                    if env(f1) == 'H'
                                        A{2}(2,f1,f2) = bb;
                                    % Adding the goal location:
                                    elseif  env(f1) == 'G'
                                        A{2}(1,f1,f2) = aa;                  
                                    else 
                                        A{2}(3,f1,f2) =1;
                                    end                                       
    end       
end

for g = 1:Ng
    a{g} = A{g}*100;
end


%--------------------------------------------------------------------------
% Transitions from t to t+1: P(S_t| S_t-1, pi)
% The B(u) matrices encode action-specific transitions
%--------------------------------------------------------------------------
for f = 1:Nf
        B{f} = e*ones(Ns(f));
end

% Actions = 4, left, down, right, up:
% Left:
B{1}(:,:,1)  =  circshift(eye(9)*10 + e, -1);
for i = [1, 4, 7]
    B{1}(:, i,1) = e;
    B{1}(i, i,1) = 10+e;
end

% down:
B{1}(:,:,2) = e*ones(Ns(1));
for i = 1:5
    B{1}(i+3, i,2) = 10+ e;
end
for i = 7:9 
    B{1}(:, i,2) = e;
    B{1}(i,i,2) = 10+e;
end

% Right: 
B{1}(:,:,3)  = circshift(eye(9)*10 + e, 1);
for i = [3,6, 9]
    B{1}(:, i,3) = e;
    B{1}(i, i,3) = 1+e;
end

% Up:
B{1}(:,:,4) = e*ones(Ns(1));
for i = 4:9
    B{1}(i-3, i,4) = 10+ e;
end
for i = 1:3 
     B{1}(:, i,4) = e;
    B{1}(i,i,4) = 10+e;
end


% Goal State: 
B{1}(:,6,1:4) = e; 
B{1}(6,6,1:4) = 100+e; 

B{1}(:,8,1:4) = e; 
B{1}(8,8,1:4) = 100+e; 


% Context, which cannot be changed by action
%--------------------------------------------------------------------------
B{2}(:,:,1)  = eye(2);

% ln(P(o))
for g = 1:Ng
    C{g}  = zeros(No(g),1);
end

C{1} = [-5 0 0 0 0 0 0 0 0]';

c= 10;
C{2}(1,:) = c;
C{2}(2,:) = -c; 

%------------------------------------------------------------------------------
% Allowable policies (of depth T).  These are just sequences of actions
% Time * Poilicy * Factor
%------------------------------------------------------------------------------

% Deep Policies:
V(:,:,1) =  [2 3 2 3 2 3 2 2 3;
                 2 3 3 2 3 2 3 3 2;
                 3 2 3 2 2 1 1 4 4];          
V(:,:,2) = 1;

 
%--------------------------------------------------------------------------
% Specify the generative model
%--------------------------------------------------------------------------
% Hyper-parameters:
%mdp.beta = exp(1);            % Precision over precision (Gamma hyperprior)   
%mdp.alpha = 64;            % Precision over action selection: default 512
%mdp.chi = exp(100);
mdp.tau   = 12;

%mdp.T = 5;
mdp.a = a;                      % Internal model
mdp.A = A;                      % Environment
mdp.b = B;                      % Internal model
mdp.B = B;                      % Environment 
mdp.C = C;                      % Preferred outcomes
mdp.D = D;                      % Prior over initial states
mdp.s = [1,2]';
% Specifying policy:
mdp.V =V; 

% Labels:
mdp.Bname = {'Position','Context'};
mdp.Aname = {'Location-A', 'Goal-A'};
mdp.label.modality = {'Location', 'Goal'};

% Checking model:
mdp         = spm_MDP_check(mdp);

return 
