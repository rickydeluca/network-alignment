import torch

from algorithms.SIGMA.function import predict


def train(model, graph_s, graph_t, idx2node_s, idx2node_t, num_nodes, cfg):

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.L2NORM)

    p_s = graph_s.x
    p_t = graph_t.x
    cost_s = graph_s.edge_index
    cost_t = graph_t.edge_index

    print('ps', p_s.shape)
    print('pt', p_t.shape)
    print('cost_s', cost_s.shape)
    print('cost_t', cost_t.shape)

    for epoch in (range(cfg.TRAIN.EPOCHS)):
        # forward model
        model.train()
        _, loss = model(p_s, cost_s, p_t, cost_t, cfg.SIGMA.T, miss_match_value=cfg.SIGMA.MISS_MATCH_VALUE)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del loss

        # evaluate
        with torch.no_grad():
            model.eval()
            logits_t, _ = model(p_s, cost_s, p_t, cost_t, cfg.SIGMA.T, miss_match_value=cfg.SIGMA.MISS_MATCH_VALUE)
            evaluate(logits_t, epoch=epoch, idx2node_s=idx2node_s, idx2node_t=idx2node_t, num_nodes=num_nodes)

    return model
    
def evaluate(log_alpha, epoch=0, idx2node_s=None, idx2node_t=None, num_nodes=None):
    matched_row, matched_col = predict(log_alpha, n=num_nodes, m=num_nodes)

    pair_names = []
    for i in range(matched_row.shape[0]):
        pair_names.append([idx2node_s[matched_row[i]], idx2node_t[matched_col[i]]])

    node_correctness = 0
    for pair in pair_names:
        if pair[0] == pair[1]:
            node_correctness += 1
    node_correctness /= num_nodes

    print('Epoch: %d, NC: %.1f' % (epoch+1, node_correctness * 100))