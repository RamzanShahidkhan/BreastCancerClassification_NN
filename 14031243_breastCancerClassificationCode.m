
r_data = importdata('breast-cancer-wisconsin.data.txt');
input_data = r_data(:,(2:10));  %separate inout data
meanValue = round(mean(input_data(:,6)));
input_data(input_data(:,6)==0) = meanValue;

output_data =r_data(:,(11)); % separate output data
output_data(output_data==2) = -1; %replace 2 with -1
output_data(output_data==4) = 1; % replace 4 with 1
        
benign_rows = 1; 
malignant_rows = 1;
training_data = []; % data for training network
training_target = []; % target data for training

testing_data = []; % data for testing
testing_target =[]; % target used for

for n=1:699
    %get benign data
   if(benign_rows<=229)
       if(output_data(n,1)==-1)
           training_data=[training_data;input_data(n,:)];
           training_target= [training_target;0];
           benign_rows =benign_rows+1;
       end
   elseif(output_data(n,1)==-1)
       testing_data=[testing_data;input_data(n,:)];
       testing_target=[testing_target;0];
   end
   % malignant data
   if(malignant_rows<=121)
       if(output_data(n,1)==1)
           training_data=[training_data;input_data(n,:)];
           training_target= [training_target;1]
           malignant_rows =malignant_rows+1;
       end
   elseif(output_data(n,1)==1)
       testing_data=[testing_data;input_data(n,:)];
       testing_target=[testing_target;1];
   end
end
 
%now make a neural network and train it
net = newff(training_data',training_target',{10,10},{'tansig','tansig','tansig'},'trainr','learngd','mse');
net.trainParam.goal=0.01;
net.trainParam.epochs=100;
net.trainParam.lr=0.02;
net= train(net,training_data',training_target'); % train the net
result=net(testing_data'); %testing net
result=round(result); %round off the values of result

accuracy = minus(testing_target',result); %subtract testing_target and result to find numbers of match during testing
match = sum(accuracy(:)==0); %correct result
mismatch=sum(accuracy(:)==1); %wrong result
accuracy_percentage = (match/length(testing_target)*100);  %dividing matched results with total number of testing data rows
disp(accuracy_percentage);  % print percentage
