function model = meta_ftm(train_doc, K, train_doc_label, w_F, alpha_option, beta_option, test_doc_A, test_doc_B, test_doc_label)


%*************************************************************************
% Matlab code for Focused Topic Model with Meta-data
% Written by He Zhao, ethanhezhao@gmail.com
% Copyright @ He Zhao
%*************************************************************************


max_iter = 20;

burn_in = 10;

thinning = 1;


N = max(train_doc(:,1));


V = size(w_F,1);

switch alpha_option
    
    
    case 'meta'
        
        d_L = size(train_doc_label, 2);
        
        d_W = 0.1*randn(K,d_L); d_c = 0.1*randn(K,1);

        alpha = 0.1 * ones(N,1);
        
    otherwise
        alpha = 0.1 * ones(N,1);
        
        
        
end

switch beta_option
    
   
        
    case 'meta'
        
        w_L = size(w_F, 2);
        
        w_W = 0.1*randn(K,w_L); w_c = 0.1*randn(K,1);
        
                
        beta_dir = 0.01 * ones(K,1);
        
    otherwise
        
       beta_dir = 0.01 * ones(K,1);
        
       sum_beta_dir = sum(beta_dir);
        
end



n_doc_topic = zeros(N,K);

n_topic_word = zeros(K,V);

n_topic_dot = zeros(K,1);

n_doc_dot = zeros(N,1);


ii = train_doc(:,1);
vv = train_doc(:,2);
mm = train_doc(:,3);

data_length = length(ii);

num_words = sum(mm);

z = zeros(data_length, max(mm));



for l = 1:data_length
    i = ii(l);
    
    v = vv(l);
    
    m = mm(l);
    
    
    for ww = 1:m
        
        k = randi(K, 1);
        
        n_doc_topic(i,k) = n_doc_topic(i,k) + 1;
        
        n_topic_word(k,v) = n_topic_word(k,v) + 1;
        
        n_doc_dot(i) = n_doc_dot(i) + 1;
        
        n_topic_dot(k) = n_topic_dot(k) + 1;
        
        z(l, ww) = k;
        
    end
    
end


if  strcmp(alpha_option, 'meta')
    d_b = zeros(N,K);
    d_b(n_doc_topic > 0) = 1;
    d_b_doc_dot = sum(d_b,2);
else
    d_b = ones(N,K);
    d_b_doc_dot = sum(d_b,2);
end


if strcmp(beta_option, 'meta')
    w_b = zeros(K,V);
    w_b(n_topic_word > 0) = 1;
    w_b_topic_dot = sum(w_b,2);
else
    w_b = ones(K,V);
    w_b_topic_dot = sum(w_b,2);
end





theta = 0;

phi = 0;

avg_count = 0;

timing= [];


for r = 1:max_iter
    
    if r > burn_in
        tic
    end
    
    tic
    
    for l = 1:data_length
        i = ii(l);
        
        v = vv(l);
        
        m = mm(l);
        
        k = z(l,:);
        
        
        if strcmp(alpha_option, 'meta')
            d_nz_k_idx = d_b(i,:)';
        else
            d_nz_k_idx = ones(K,1);
        end
        
        if strcmp(beta_option, 'meta')
            w_nz_k_idx = w_b(:,v);
        else
            w_nz_k_idx = ones(K,1);
        end
        
        nz_k_idx = logical(d_nz_k_idx .* w_nz_k_idx);
        
        
        for ww = 1:m
            
            k_ww = k(ww);
            
            n_doc_topic(i,k_ww) = n_doc_topic(i,k_ww) - 1;
            n_topic_word(k_ww,v) = n_topic_word(k_ww,v) - 1;
            n_topic_dot(k_ww) = n_topic_dot(k_ww) - 1;
            
            
            p_left = (alpha(i) + n_doc_topic(i,:));
            p_right = zeros(K,1);
            if strcmp(beta_option,'meta')
                p_right(nz_k_idx) = (beta_dir(nz_k_idx) + n_topic_word(nz_k_idx,v)) ./ ...
                    (beta_dir(nz_k_idx) .* w_b_topic_dot(nz_k_idx) + n_topic_dot(nz_k_idx));
            else
                p_right = (beta_dir + n_topic_word(:,v)) ./ ...
                    (sum_beta_dir + n_topic_dot);
            end
            
            
            p = (p_left .* p_right') .* nz_k_idx';
            
            sum_cum = cumsum(p(:));
            
            new_k_m = find(sum_cum > rand() * sum_cum(end),1);
            
            n_doc_topic(i,new_k_m) = n_doc_topic(i,new_k_m) + 1;
            n_topic_word(new_k_m,v) = n_topic_word(new_k_m,v) + 1;
            
            n_topic_dot(new_k_m) = n_topic_dot(new_k_m) + 1;
            
            z(l,ww) = new_k_m;
            
            
        end
        
        
    end
    
    if strcmp(alpha_option, 'meta')
        
        
        H2train = train_doc_label';
        
        Ntrain = N;
        
        K1 = K;
        
        K2 = d_L;
        
        Xmat = bsxfun(@plus,d_W*H2train,d_c); % K1*n
        Xmat = 1 ./ (1 + exp(-Xmat'));

        
        
        for k = 1:K
            i =  n_doc_topic(:,k) == 0;
            d_b_doc_dot(i) = d_b_doc_dot(i)-d_b(i,k);
            temp = (d_b_doc_dot(i)) .* alpha(i);
            p_1 = exp(gammaln(temp+n_doc_dot(i)) - gammaln(temp+n_doc_dot(i) + alpha(i)) + ...
            gammaln(temp + alpha(i)) - gammaln(temp)) .* Xmat(i,k);
            d_b(i,k) = rand(nnz(i),1) < (p_1 ./ (p_1 + 1 - Xmat(i,k)));
            d_b_doc_dot(i) = d_b_doc_dot(i) + d_b(i,k);            
        end
        
        
        H1train = d_b';
        Xmat = bsxfun(@plus,d_W*H2train,d_c); % K1*n

        Xvec = reshape(Xmat,K1*Ntrain,1);
        gamma0vec = PolyaGamRndTruncated(ones(K1*Ntrain,1),Xvec,20);
        gamma0Train = reshape(gamma0vec,K1,Ntrain);
        
        for j = 1:K1
            Hgam = bsxfun(@times,H2train,gamma0Train(j,:));
            invSigmaW = eye(K2) + Hgam*H2train';
            MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(j,:)-0.5-d_c(j)*gamma0Train(j,:)),2));
            R = choll(invSigmaW);
            d_W(j,:) = (MuW + R\randn(K2,1))';
        end
        sigmaC = 1./(sum(gamma0Train,2)+1);
        muC = sigmaC.*sum(H1train-0.5-gamma0Train.*(d_W*H2train),2);
        d_c = normrnd(muC,sqrt(sigmaC));
        

        
    end
    
    if strcmp(beta_option, 'meta')
        
        
        H2train = w_F';
        
        Ntrain = V;
        
        K1 = K;
        
        K2 = w_L;
        
        Xmat = bsxfun(@plus,w_W*H2train,w_c); % K * V
        Xmat = 1 ./ (1 + exp(-Xmat));


        for v = 1:V
            k = n_topic_word(:,v) == 0;
            
            w_b_topic_dot(k) = w_b_topic_dot(k) - w_b(k,v);
            temp = w_b_topic_dot(k) .* beta_dir(k);
            p_1 = exp( gammaln(temp+n_topic_dot(k)) - gammaln(temp+n_topic_dot(k) + beta_dir(k)) ...
            + gammaln(temp + beta_dir(k)) - gammaln(temp)) .* Xmat(k,v);
            w_b(k,v) = rand(nnz(k),1) < (p_1 ./ (p_1 + 1 - Xmat(k,v)));
            w_b_topic_dot(k) = w_b_topic_dot(k) + w_b(k,v);
            
        end
        
        H1train = w_b;
        Xmat = bsxfun(@plus,w_W*H2train,w_c); % K * V

        Xvec = reshape(Xmat,K1*Ntrain,1);
        gamma0vec = PolyaGamRndTruncated(ones(K1*Ntrain,1),Xvec,20);
        gamma0Train = reshape(gamma0vec,K1,Ntrain);
        
        if w_L > 1
            for j = 1:K1
                Hgam = bsxfun(@times,H2train,gamma0Train(j,:));
                invSigmaW = eye(K2) + Hgam*H2train';
                MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(j,:)-0.5-w_c(j)*gamma0Train(j,:)),2));
                R = choll(invSigmaW);
                w_W(j,:) = (MuW + R\randn(K2,1))';
            end
        end
        sigmaC = 1./(sum(gamma0Train,2)+1);
        muC = sigmaC.*sum(H1train-0.5-gamma0Train.*(w_W*H2train),2);
        w_c = normrnd(muC,sqrt(sigmaC));
        
        
        
    end
    
    



    
    

    
    toc
    if r > burn_in
        timing(end+1) = toc;
    end
    
    if r > burn_in && mod(r - burn_in,thinning) == 0
        
            
        temp_theta = zeros(N,K);

        for i = 1:N
            d_nz_k_idx = find(d_b(i,:));
            temp_theta(i,d_nz_k_idx) = (alpha(i) + n_doc_topic(i,d_nz_k_idx)) ./ ( d_b_doc_dot(i) .* alpha(i) + sum(n_doc_topic(i,d_nz_k_idx)));

        end
        

        
        
            
        temp_phi = zeros(K,V);

        for k = 1:K
            w_nz_k_idx = logical(w_b(k,:));
            temp_phi(k,w_nz_k_idx) = (beta_dir(k) + n_topic_word(k,w_nz_k_idx)) ./ (beta_dir(k) * w_b_topic_dot(k) + n_topic_dot(k));

        end
            


        theta = theta + temp_theta;
        phi = phi + temp_phi;
        avg_count = avg_count + 1;
    end
    
    if mod(r,10) == 0
        

        temp_theta = zeros(N,K);

        for i = 1:N
            d_nz_k_idx = find(d_b(i,:));
            temp_theta(i,d_nz_k_idx) = (alpha(i) + n_doc_topic(i,d_nz_k_idx)) ./ ( d_b_doc_dot(i) .* alpha(i) + sum(n_doc_topic(i,d_nz_k_idx)));

        end

        


        temp_phi = zeros(K,V);

        for k = 1:K
            w_nz_k_idx = logical(w_b(k,:));
            temp_phi(k,w_nz_k_idx) = (beta_dir(k) + n_topic_word(k,w_nz_k_idx)) ./ (beta_dir(k) * w_b_topic_dot(k) + n_topic_dot(k));

        end

        
        prob = sum(temp_theta(ii,:) .* temp_phi(:,vv)',2);
        pp = log(prob) .* mm;
        
        pp = exp(-sum(pp) ./ num_words);
        
        fprintf('iter %d, train-pp: %d\n', r, pp);

    end
    
end

theta = theta ./avg_count;

phi = phi ./avg_count;


model.alpha = alpha;

model.beta_dir = beta_dir;

model.theta = theta;

model.phi = phi;

model.n_doc_topic = sparse(n_doc_topic);

model.n_topic_word = sparse(n_topic_word);

model.timing = timing;

model.alpha_option = alpha_option;

model.beta_option = beta_option;


if strcmp(alpha_option,'meta')
    model.d_W = d_W;
    model.d_c = d_c;
    model.d_b = d_b;
end

if strcmp(beta_option,'meta')
    model.w_W = w_W;
    model.w_c = w_c;
    model.w_b = w_b;
end


if ~isempty(test_doc_A)
    model = meta_ftm_test(test_doc_A, test_doc_B, test_doc_label, model);
end

end





