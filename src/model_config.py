CNN_dic = {
    'ss1' : {
        'l1_numFilters' : 128,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    },

    'ss2' : {
        'l1_numFilters' : 128,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    },
    'har1' : {
        'l1_numFilters' : 128,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    },
    'har2' : {
        'l1_numFilters' : 128,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    },
    'sn1' : {
        'l1_numFilters' : 64,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    },
    'sn2' : {
        'l1_numFilters' : 64,
        'l2_numFilters' : 64,
        'l1_kernelSize' : 32,
        'l2_kernelSize' : 16,
        'l1_maxPoolSize' : 4,
        'l2_maxPoolSize' : 4,
        'dropout' : 0.25
    }
}

LSTM_dic = {
    'ss1' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    },
    'ss2' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    },
    'har1' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    },
    'har2' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    },
    'sn1' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    },
    'sn2' : {
        'lstm_units' : 16,
        'dropout' : 0.25,
        'hidden_dense_size' : 128
    }
}

Transf_dic = {
    'ss1' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
    'ss2' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
    'har1' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
    'har2' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
    'sn1' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
    'sn2' : {
        'num_attn+layers' : 2,
        'head_size' : 16,
        'num_heads' : 4,
        'ff_dim' : 32,
        'dropout' : 0.25
    },
}

def loadDic(model):
    if model=='CNN':
        return dict(CNN_dic)
    elif model=='LSTM':
        return dict(LSTM_dic)
    elif model=='Transformer':
        return dict(Transf_dic)
    else:
        print('Unrecognized model name in loadDic')

