import torch
import torch.nn as nn

class RandomNormal(object):
    """Initialize embedding with a normal distribution.
    Args:
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        return embed


class SequenceFeature(object):
    """
    The Feature Class for Sequence feature or multi-hot feature.
    If you use this feature, you must pad the feature value.
    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        pooling (str): pooling method, support `["mean", "sum", "concat"]` (default=`"mean"`)
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(self,
                 name,
                 vocab_size,
                 embed_dim=None,
                 pooling="mean",
                 shared_with=None,
                 padding_idx=None,
                 initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.pooling = pooling
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SequenceFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed


class SparseFeature(object):
    """
    Args:
        name (str): feature's name.
        vocab_size (int): vocabulary size of embedding table.
        embed_dim (int): embedding vector's length
        shared_with (str): the another feature name which this feature will shared with embedding.
        padding_idx (int, optional): If specified, the entries at padding_idx will be masked 0 in InputMask Layer.
        initializer(Initializer): Initializer the embedding layer weight.
    """

    def __init__(self, name, vocab_size, embed_dim=None, shared_with=None, padding_idx=None, initializer=RandomNormal(0, 0.0001)):
        self.name = name
        self.vocab_size = vocab_size
        if embed_dim is None:
            self.embed_dim = get_auto_embedding_dim(vocab_size)
        else:
            self.embed_dim = embed_dim
        self.shared_with = shared_with
        self.padding_idx = padding_idx
        self.initializer = initializer

    def __repr__(self):
        return f'<SparseFeature {self.name} with Embedding shape ({self.vocab_size}, {self.embed_dim})>'

    def get_embedding_layer(self):
        if not hasattr(self, 'embed'):
            self.embed = self.initializer(self.vocab_size, self.embed_dim)
        return self.embed


class DenseFeature(object):
    """
    Args:
        name (str): feature's name.
        embed_dim (int): embedding vector's length, the value fixed `1`.
    """

    def __init__(self, name):
        self.name = name
        self.embed_dim = 1

    def __repr__(self):
        return f'<DenseFeature {self.name}>'

    
def get_auto_embedding_dim(num_classes):
    """ 
    Calculate the dim of embedding vector according to the number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    
    Args:
        num_classes: number of classes in the category
    
    Returns:
        the dim of embedding vector
    """
    return np.floor(6 * np.pow(num_classes, 0.26))