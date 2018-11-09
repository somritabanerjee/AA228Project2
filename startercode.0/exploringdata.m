%% AA 228 Project 2 
% Somrita Banerjee
clc
clear all
close all
small_data = csvread('small.csv',1,0);
medium_data = csvread('medium.csv',1,0);
large_data = csvread('large.csv',1,0);
figure
plot(small_data(:,1),small_data(:,2),'o')
title('Small')
xlabel('Current state');
ylabel('Action taken');
figure
plot(small_data(:,4),small_data(:,3),'o')
title('Small')
xlabel('Transitioned state');
ylabel('Reward received');

%% Medium
figure
plot(medium_data(:,4),medium_data(:,3),'o')
title('Medium')
xlabel('Transitioned state');
ylabel('Reward received');
figure
plot(medium_data(:,1),medium_data(:,4),'o')
title('Medium')
ylabel('Transitioned state');
xlabel('Initial state');
figure
plot(medium_data(:,1),medium_data(:,4),'o')
title('Medium')
ylabel('Transitioned state');
xlabel('Initial state');
ylim([2*10^4 3*10^4])

figure
ind=find(medium_data(:,3)>0)
plot(medium_data(ind,4),medium_data(ind,3),'o')
title('Medium')
xlabel('Transitioned state');
ylabel('Reward received');

figure
ind=find(medium_data(:,3)>0)
plot(medium_data(ind,1),medium_data(ind,4),'o')
title('Medium')
xlabel('Initial state');
ylabel('Transitioned state');

unique_final_states_medium=unique(medium_data(:,4));
% Only 500 unique final states!
unique_initial_states_medium=unique(medium_data(:,1));

%% Large
s=large_data(:,1);
a=large_data(:,2);
r=large_data(:,3);
sp=large_data(:,4);
figure
plot(sp,r,'o')
title('Large')
xlabel('Transitioned state');
ylabel('Reward received');

unique_final_states=unique(sp);
% Only 500 unique final states!
unique_initial_states=unique(s);
% The same 500 unique initial states! 
dictionary=[(1:500)',unique_initial_states];
final_states_new_numbering=zeros(length(large_data),1);
initial_states_new_numbering=zeros(length(large_data),1);
for i=1:length(large_data)
    final_states_new_numbering(i) = find(unique_final_states==sp(i));
    initial_states_new_numbering(i) = find(unique_initial_states==s(i));
end
R=large_data(:,3);
figure
plot(final_states_new_numbering,R,'o')
title('Large')
xlabel('Transitioned state');
ylabel('Reward received');
ind_greater_than_20_reward=find(R>20);
statesWithG20RewardON=unique(sp(ind_greater_than_20_reward));
% [150211;150413;151203]
statesWithG20RewardNN=unique(final_states_new_numbering(ind_greater_than_20_reward));
% Only 16, 38, 63
rewardsG20= unique(R(ind_greater_than_20_reward));
% Only 100 (at 16) , 200 (at 38, 63)
figure
plot(final_states_new_numbering(ind_greater_than_20_reward),R(ind_greater_than_20_reward),'o')
title('Large - only if reward > 20')
xlabel('Transitioned state (new numbering)');
ylabel('Reward received');
ylim([0 210])
text(16+1,100,'sp=16, r=100')
text(38+1,200,'sp=38, r=200')
text(63-8,200-8,'sp=63, r=200')
figure
plot(sp(ind_greater_than_20_reward),R(ind_greater_than_20_reward),'o')
title('Large - only if reward > 20')
xlabel('Transitioned state');
ylabel('Reward received');
ylim([0 210])
text(150211+50,100,'sp=150211, r=100')
text(150413+50,200,'sp=150413, r=200')
text(151203-200,200-8,'sp=151203, r=200')

ind_greater_than_0_reward=find(R>0);
finalStatesWithG0RewardON=unique(sp(ind_greater_than_0_reward));
% 11 states [150211;150413;151203;230402;231014;270203;271310;290213;291112;300311;301103]
finalStatesWithG0RewardNN=unique(final_states_new_numbering(ind_greater_than_0_reward));
% Only [16;38;63;132;149;213;275;318;357;426;453]
rewardsG0= unique(R(ind_greater_than_0_reward));
% Only 5, 10, 100, 200
figure
plot(final_states_new_numbering(ind_greater_than_0_reward),R(ind_greater_than_0_reward),'o')
title('Large - only if reward > 0')
xlabel('Transitioned state (new numbering)');
ylabel('Reward received');
ylim([0 230])
text(16+10,100,'sp=16, r=100')
text(38-20,200+10,'sp=38, r=200')
text(63,200-10,'sp=63, r=200')
figure
plot(initial_states_new_numbering(ind_greater_than_0_reward),R(ind_greater_than_0_reward),'o')
title('Large - only if reward > 0')
xlabel('Initial state (new numbering)');
ylabel('Reward received');
ylim([0 230])

% figure 
% plot(s,a,sp)
figure
plot(s,sp,'o')
title('Large')
ylabel('Transitioned state s''');
xlabel('Current state s');
figure
plot(initial_states_new_numbering,final_states_new_numbering,'o')
title('Large')
ylabel('Transitioned state s'' (new numbering)');
xlabel('Current state s (new numbering)');

figure
for actionIdx=1:9
    indices=find(a==actionIdx);
    subplot(3,3,actionIdx); 
    plot(initial_states_new_numbering(indices),final_states_new_numbering(indices),'o');
    title(['action = ',num2str(actionIdx)]);
    ylabel('s''');
    xlabel('s');
    xlim([0 550]);
    ylim([0 550]);
end

figure
for actionIdx=1:9
    indices=find(a==actionIdx);
    subplot(3,3,actionIdx); 
    plot(initial_states_new_numbering(indices),r(indices),'o');
    title(['action = ',num2str(actionIdx)]);
    ylabel('reward');
    xlabel('s');
    xlim([0 550]);
%     ylim([0 550]);
end

%% Actions analysis
for actionIdx=1:9
    lower=90;
    upper=110;
    m=initial_states_new_numbering;
    indices=find((not(abs(sign(sign(lower-m)+sign(upper-m))))+ (a==actionIdx))>1);
    subplot(3,3,actionIdx); 
    plot(initial_states_new_numbering(indices),final_states_new_numbering(indices),'o');
    title(['action = ',num2str(actionIdx)]);
    ylabel('s''');
    xlabel('s');
    xlim([lower-10 upper+10]);
    ylim([lower-10 upper+10]);
end

figure
for actionIdx=1:9
    lower=50;
    upper=200;
    m=initial_states_new_numbering;
    indices=find((not(abs(sign(sign(lower-m)+sign(upper-m))))+ (a==actionIdx))>1);
    subplot(3,3,actionIdx); 
    plot(initial_states_new_numbering(indices),final_states_new_numbering(indices),'o');
    title(['action = ',num2str(actionIdx)]);
    ylabel('s''');
    xlabel('s');
    xlim([lower-10 upper+10]);
    ylim([lower-10 upper+10]);
end

%% others

figure
for actionIdx=1:9
    m=initial_states_new_numbering;
    indices=find((not(abs(sign(sign(90-m)+sign(110-m))))+ (a==actionIdx))>1);
    subplot(3,3,actionIdx); 
    plot(initial_states_new_numbering(indices),r(indices),'o');
    title(['action = ',num2str(actionIdx)]);
    ylabel('reward');
    xlabel('s');
%     xlim([0 550]);
%     ylim([0 550]);
end


figure
plot(initial_states_new_numbering(ind_greater_than_20_reward),final_states_new_numbering(ind_greater_than_20_reward),'o')
hold on
plot([1:100],[1:100],'--')
title('Transitions leading to r>20')
ylabel('s''');
xlabel('s');

figure
plot(initial_states_new_numbering(ind_greater_than_20_reward),a(ind_greater_than_20_reward),'o')
title('Actions leading to r>20')
ylabel('a');
xlabel('s');

MatrixG20NN=[initial_states_new_numbering(ind_greater_than_20_reward) a(ind_greater_than_20_reward) r(ind_greater_than_20_reward) final_states_new_numbering(ind_greater_than_20_reward)];
unique(MatrixG20NN,'rows')
MatrixG20ON=[s(ind_greater_than_20_reward) a(ind_greater_than_20_reward) r(ind_greater_than_20_reward) sp(ind_greater_than_20_reward)];
unique(MatrixG20ON,'rows')

% ind=find(initial_states_new_numbering>=90 && initial_states_new_numbering<=110)
m=initial_states_new_numbering;
ind=find((not(abs(sign(sign(90-m)+sign(110-m))))+ (a==4))>1);
MatrixCloseTo100NN=[initial_states_new_numbering(ind) a(ind) r(ind) final_states_new_numbering(ind)];
figure
plot(MatrixCloseTo100NN(:,1),MatrixCloseTo100NN(:,4),'o')
hold on
plot([90:110],[90:110],'--');
plot([90:110],[100:120],'--');
title('Behavior near 100 for a==4');

ind=find((not(abs(sign(sign(90-m)+sign(110-m))))+ (a==3))>1);
MatrixCloseTo100NN=[initial_states_new_numbering(ind) a(ind) r(ind) final_states_new_numbering(ind)];
figure
plot(MatrixCloseTo100NN(:,1),MatrixCloseTo100NN(:,4),'o')
hold on
plot([90:110],[90:110],'--'); %sp=s
plot([90:110],[80:100],'--'); %sp=s-10
title('Behavior near 100 for a==3');

% Policy:
% For the 312020 - 500 states, just set action to any one of 5-9 doesn't
% matter. Let's just say we set all of them to 9. 
% for the 500 that we do see, 
% (s,9) for all s not in unique_final_states
% If we're in the states with 200 reward, stay there! Set action to 1.
% (150413,1)
% (151203,1)
% At 150211, reward is 100. 