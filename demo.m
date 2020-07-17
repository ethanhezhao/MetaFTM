


load('WS.mat');

K = 50;

alpha_option = 'off'; % use anything else to turn focusing off

beta_option = 'meta'; % use anything else to turn focusing off


model = meta_ftm(train_doc, K, train_doc_label, word_embeddings, alpha_option, beta_option, test_doc_A, test_doc_B, test_doc_label);


save_dir = './save';

if ~exist(save_dir,'dir')
    mkdir(save_dir);
end

save(sprintf('%s/model.mat',save_dir),'model');

top_words_file = fopen(sprintf('%s/top_words.txt',save_dir),'w');

show_top_words_simple(model.phi, voc, 20, top_words_file);

fclose(top_words_file);
