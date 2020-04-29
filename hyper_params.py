# args = {
#     'gpu_num': 7,
#
#     'dataset_name': 'dev_test',
#     'experiment_name': 'hint_sst_2_class',
#
#     #pre-trained embedding
#     'embedding_loc': 'glove',
#     'embedding_dim': 300,
#
#     #sequential data
#     'seq_length': 63,
#     'z_length': 194,
#     'batch_size': 32,
#
#     #transformer
#     'transformer_layers': 1,
#     'num_heads': 4,
#
#     'sequence_attn_dim': 300,
#     'lex_attn_dim': None, #set to None if no lex
#     'feed_forward_dim': 300,
#
#     #general
#     'alpha': 1.0,
#     'output_dim': 2,
#     'dropout': 0.3,
#     'warmup_steps': 1000, #see 'Attention is all you need'
#     'max_epoch': 100,
#     'tolerance': 10,
#     'lr': 0.1
# }
args = {
            'dataset_name': 'dev_test',
            'experiment_name': 'hint_global_repeat',

            # pre-trained embedding
            'embedding_loc': 'glove',
            'embedding_dim': 300,

            # sequential data
            'seq_length': 63,
            'z_length': 194,
            'mask_ratio': 0.5,
            'batch_size': 64,

            # transformer
            'transformer_layers': 2,
            'num_heads': 4,

            'sequence_attn_dim': 300,
            'lex_attn_dim': 300,  # set to None if no lex
            'feed_forward_dim': 300,

            # general
            'alpha': 0.7,
            'output_dim': 2,
            'dropout': 0.3,
            'max_epoch': 100,
            'tolerance': 10,
            'lr': 0.1
        }