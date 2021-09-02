clear all; 
clc;
%% 加载数据
train_data=xlsread('train_data');                           % ART1网络输入为二进制,train_data为七个样本，每一列为一个样本
data_length=size(train_data,1);
data_num=size(train_data,2);
N=100;

%% 网络参数初始化
R_node_num=3;
weight_b=ones(data_length,R_node_num)/N;
weight_t=ones(data_length,R_node_num);
threshold_ro=0.5;

%%
%EM网络初始化
t = 0.1; %时间衰减系数 
em_train_data_1 = ones(1 , 6) ; %建立7个事件的短期记忆
em_train_data = ones(1 , 6) ; %建立7个事件的短期记忆
em_data_length = size(em_train_data , 1); %情景记忆层输入数据初始化
em_data_num = size(em_train_data , 2);
N = 100;
em_node_num = 1;
em_weight_b=ones(6,em_node_num)/N;
em_weight_t=ones(6,em_node_num);
em_threshold = 0.5 ;
EM_ART_activation = ones(1 , 6);  

%% 开始网络训练
for i=1:data_num

        R_node=zeros(R_node_num,1);                 %%匹配循环过程标志
            for n=1:R_node_num                          %%匹配循环过程
                        %寻找竞争获胜神经元
                            for j=1:R_node_num
                                net(j,1)=sum(train_data(:,i).*weight_b(:,j));
                            end
                             [~,j_max]=max(net);
                             if R_node(j_max,1)==1                  %%循环激活标志判断
                                 net(j_max,1)= -n;
                             end
                              [~,j_max]=max(net);
                              R_node(j_max,1)=1;  %%去激活
                                      %竞争获胜神经元通过外星权向量返回C层，进行相似度计算
            weight_t_active=weight_t(:,j_max);
            weight_b_active=weight_b(:,j_max);
            Similarity_N0=sum(weight_t_active.*train_data(:,i));
            Similarity_N1=sum(train_data(:,i));
            if (threshold_ro<Similarity_N0/Similarity_N1)
                [weight_t(:,j_max),weight_b(:,j_max)]=ART1_learn(train_data(:,i),weight_t_active,weight_b_active);
                fprintf('样本%d属于第%d类\n',i,j_max);
                flag=0;
            break;
            end
            flag=1;
            end
            %% 判断是否需要添加新的结点，更新网络
    if(flag==1)
%       ART1_updata_model()
        R_node_num=R_node_num+1;
        fprintf('样本%d属于第%d类\n',i,R_node_num);
        weight_b=[weight_b,train_data(:,j)];
        weight_t=[weight_t,ones(data_length,1)];
    end
%%
 %emART层记录
    em_train_data_temp = em_train_data_1(: , 2:6);
    em_train_data_1(: , 1:5) = em_train_data_temp;
    em_train_data_1(: , 6) = j_max ;
    em_train_data = em_train_data_1 /10  ; %情景记忆层时间衰减序列 .* em_ART_t.'
    
    %EMART神经网络
    %EM_ART_node = zeros(EMART_node_num , 1);
    em_node=zeros(em_node_num,1);                 %%匹配循环过程标志
            for n=1:em_node_num                          %%匹配循环过程
                        %寻找竞争获胜神经元
                            for j=1:em_node_num
                                em_net(: , j)=sum(em_train_data * em_weight_b(:,j));
                            end
                             [~,em_max]=max(em_net);
                             if em_node(em_max,1)==1                  %%循环激活标志判断
                                 em_net(em_max,1)=-n;
                             end
                              [~,em_max]=max(em_net);
                              R_node(em_max,1)=1;  %%去激活
                                      %竞争获胜神经元通过外星权向量返回C层，进行相似度计算
            em_weight_t_active=em_weight_t(:,em_max);
            em_weight_b_active=em_weight_b(:,em_max);
            em_Similarity_N0=sum(sum(em_weight_t_active .* em_train_data));
            em_Similarity_N1=sum(em_train_data);
            b = em_Similarity_N0/em_Similarity_N1;
            if (em_threshold < b)
                %[em_weight_t(:,em_max), em_weight_b(: , em_max) ]= ART1_learn(em_train_data.' ,em_weight_t_active,em_weight_b_active);
                em_weight_t(:,em_max) = em_train_data.' .* em_weight_t_active;
                em_weight_b(: , em_max) = em_weight_t(:,em_max) ./  (0.5+sum(em_weight_t(:,em_max)));
                fprintf('样本%d属于第%d序列\n',i,em_max);
                flag=0;
            break;
            end
            flag=1;
            end
            %% 判断是否需要添加新的结点，更新网络
    if(flag==1)
%       ART1_updata_model()
        em_node_num=em_node_num+1;
        fprintf('样本%d属于第%d序列\n',i,em_node_num);
        em_weight_b=[em_weight_b , em_train_data.'];
        em_weight_t=[em_weight_t , ones(6,1)];
    end
    
  
end




