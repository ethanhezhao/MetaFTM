function train_model = meta_ftm_test(test_doc_A, test_doc_B, test_doc_label, train_model)

%*************************************************************************
% Matlab code for Focused Topic Model with Meta-data
% Written by He Zhao, ethanhezhao@gmail.com
% Copyright @ He Zhao
%*************************************************************************

max_iter  = 500;

burn_in  = 300;

thinning = 10;

test_A_N = max(test_doc_A(:,1));


[K,V] = size(train_model.phi);

alpha_option = train_model.alpha_option;

train_phi = train_model.phi;




switch alpha_option


    
    case 'meta'
        
        d_F = test_doc_label;
        
        
       
        
        Xmat = bsxfun(@plus,train_model.d_W*d_F',train_model.d_c); % K1*test_A_N
        Xmat = 1 ./ (1 + exp(-Xmat'));

end



alpha = train_model.alpha(1) .* ones(test_A_N,1);





n_doc_topic = zeros(test_A_N,K);



n_doc_dot = zeros(test_A_N,1);


ii = test_doc_A(:,1);
vv = test_doc_A(:,2);
mm = test_doc_A(:,3);

data_length = length(ii);


z = zeros(data_length, max(mm));



% unseen_words = find(sum(train_phi,1)==0);

unseen_words = find(sum(train_model.n_topic_word,1)==0);


seen_pairs = true(data_length,1);


for l = 1:data_length
    i = ii(l);
    
    v = vv(l);
    m = mm(l);
    
    if ~isempty(find(unseen_words == v, 1))
        seen_pairs(l) = 0;
        continue;
    end

    active_k = find(train_phi(:,v));
    if isempty(active_k)
        active_k = 1:K;
    end
    for ww = 1:m
     
        k = randi(length(active_k), 1);
        k = active_k(k);              
        n_doc_topic(i,k) = n_doc_topic(i,k) + 1;
                
        n_doc_dot(i) = n_doc_dot(i) + 1;        
        
        z(l, ww) = k;

    end

end


if strcmp(alpha_option, 'meta')
    d_b = zeros(test_A_N,K);
    d_b(n_doc_topic > 0) = 1;
    d_b_doc_dot = sum(d_b,2);
else
    d_b = ones(test_A_N,K);
    d_b_doc_dot = sum(d_b,2);
end





test_theta = 0;

avg_count = 0;

for r = 1:max_iter
    
   for l = 1:data_length
        i = ii(l);

        v = vv(l);
        
                
        if ~seen_pairs(l)
            continue;
        end

        m = mm(l);

        k = z(l,:);

        if strcmp(alpha_option, 'meta')
            d_nz_k_idx = logical(d_b(i,:)');
        else
            d_nz_k_idx = ones(K,1);
        end
        
        
        for ww = 1:m

            k_ww = k(ww);

            n_doc_topic(i,k_ww) = n_doc_topic(i,k_ww) - 1;


            p_left = (alpha(i) + n_doc_topic(i,:));
            
            p_right = train_phi(:,v);
            
            
            p = (p_left .* p_right') .* d_nz_k_idx';

            
            sum_cum = cumsum(p(:));
            
            new_k_m = find(sum_cum > rand() * sum_cum(end),1);
            n_doc_topic(i,new_k_m) = n_doc_topic(i,new_k_m) + 1;
            
            z(l,ww) = new_k_m;


        end


   end
    
    
    
    switch alpha_option


        case 'meta'
            
        

        for k = 1:K
            i =  n_doc_topic(:,k) == 0;
            d_b_doc_dot(i) = d_b_doc_dot(i)-d_b(i,k);
            temp = (d_b_doc_dot(i)) .* alpha(i);
            p_1 = exp(gammaln(temp+n_doc_dot(i)) - gammaln(temp+n_doc_dot(i) + alpha(i)) + ...
            gammaln(temp + alpha(i)) - gammaln(temp)) .* Xmat(i,k);
            d_b(i,k) = rand(nnz(i),1) < (p_1 ./ (p_1 + 1 - Xmat(i,k)));
            d_b_doc_dot(i) = d_b_doc_dot(i) + d_b(i,k);            
        end



    end
    
    
    
    if r > burn_in && mod(r-burn_in,thinning) == 0
        

            
        temp_theta = zeros(test_A_N,K);

        for i = 1:test_A_N
            nz_k_idx = find(d_b(i,:));
            temp_theta(i,nz_k_idx) = (alpha(i) + n_doc_topic(i,nz_k_idx)) ./ ( d_b_doc_dot(i) .* alpha(i) + sum(n_doc_topic(i,nz_k_idx)));

        end

        test_theta = test_theta + temp_theta;
        avg_count = avg_count + 1;

    end
    

end

seen_pairs_B = true(size(test_doc_B,1),1);

for l = 1:size(test_doc_B,1)
    v = test_doc_B(l,2);
    if ~isempty(find(unseen_words == v, 1))
        seen_pairs_B(l) = 0;
        continue;
    end
end

ii_B = test_doc_B(seen_pairs_B,1);
vv_B = test_doc_B(seen_pairs_B,2);
mm_B = test_doc_B(seen_pairs_B,3);

num_words_B = sum(mm_B);


test_theta = test_theta ./ avg_count;

prob = sum(test_theta(ii_B,:) .* train_phi(:,vv_B)',2);
pp = log(prob) .* mm_B;

pp = exp(-sum(pp) ./ num_words_B);

train_model.test_theta = test_theta;
train_model.test_pp = pp;

end


