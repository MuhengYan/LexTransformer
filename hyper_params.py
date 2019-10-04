args = {
    'dataset_name': 'dev_test',
    'experiment_name': 'sst_2_class',
    
    #pre-trained embedding
    'embedding_loc': 'sst_google',
    'embedding_dim': 300,
    
    #sequential data
    'seq_length': 50,
    'batch_size': 16,
    
    #transformer
    'transformer_layers': 3,
    'num_heads': 8,
    
    'sequence_attn_dim': 150,
    'lex_attn_dim': None, #set to None if no lex
    'feed_forward_dim': 150,

    #general
    'alpha': 0.5,
    'output_dim': 2,
    'dropout': 0.1,
    'warmup_steps': 4000, #see 'Attention is all you need'
    'max_epoch': 30,
    'tolerance': 5
}
