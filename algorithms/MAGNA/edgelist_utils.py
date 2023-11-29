def read_align_dict(edgelist_path, sep=" ", reverse=False):
    align_dict = {}
    weighted = False

    with open(edgelist_path, 'r') as ef:
        for line in ef:
            nodes = line.strip().split(sep)
            
            if len(nodes) == 2:
                source, target = nodes

                if reverse:
                    align_dict[target] = source
                else:
                    align_dict[source] = target

            if len(nodes) == 3:
                weighted = True
                source, target, weight = nodes

                if reverse:
                    align_dict[target] = (source, weight)
                else:
                    align_dict[source] = (target, weight)

    return align_dict, weighted
