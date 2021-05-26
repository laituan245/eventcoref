import torch

def get_span_emb(context_features, span_starts, span_ends):
    num_tokens = context_features.size()[0]

    features = []
    for s, e in zip(span_starts, span_ends):
        sliced_features = context_features[s:e, :]
        features.append(torch.mean(sliced_features, dim=0, keepdim=True))
    features = torch.cat(features, dim=0)
    return features

def get_pair_embs(event_features):
    n, d = event_features.size()
    features_list = []

    # Compute diff_embs and prod_embs
    src_embs = event_features.view(1, n, d).repeat([n, 1, 1])
    target_embs = event_features.view(n, 1, d).repeat([1, n, 1])
    prod_embds = src_embs * target_embs

    # Update features_list
    features_list.append(src_embs)
    features_list.append(target_embs)
    features_list.append(prod_embds)

    # Concatenation
    pair_embs = torch.cat(features_list, 2)

    return pair_embs

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.
    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
