args = {
    'gpu_num': 7,

    'dataset_name': 'dev_test',
    'experiment_name': 'sst_2_class',
    
    #pre-trained embedding
    'embedding_loc': 'sst_wiki_600',
    'embedding_dim': 600,
    
    #sequential data
    'seq_length': 57,
    'batch_size': 32,
    
    #transformer
    'transformer_layers': 1,
    'num_heads': 1,
    
    'sequence_attn_dim': 600,
    'lex_attn_dim': None, #set to None if no lex
    'feed_forward_dim': 150,

    #general
    'alpha': 1.0,
    'output_dim': 2,
    'dropout': 0.3,
    'warmup_steps': 300, #see 'Attention is all you need'
    'max_epoch': 100,
    'tolerance': 20
}
