import codecs, json, re, networkx as nx
from networkx.drawing import nx_agraph
import pygraphviz as pg
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model


path = '/Users/mayankkejriwal/datasets/memex/domain-discovery/christopher/'


def build_undirected_graph_from_divergence_file(input = path+'crawlDivergenceNYU.jl', ignore_empty_string=True,
                                                ascii_only=False):
    obj = json.load(codecs.open(input, 'r', 'utf-8'))
    empty_string = False
    G = nx.Graph()
    for k, v in obj.items():
        # print k, v
        new_k = k
        if k == '' or k == ' ':
            empty_string = True
            print obj[k]
            if ignore_empty_string:
                continue
            else:
                new_k='empty_string'
        if ascii_only:
            new_k = ''.join([i if ord(i) < 128 else '' for i in new_k])
        for k1, v1, in v.items():

            if k1 == '' or k1 == ' ':
                empty_string = True

                if ignore_empty_string:
                    continue
                else:

                    G.add_edge(new_k, 'empty_string')
            else:
                if ascii_only:
                    k1 = ''.join([i if ord(i) < 128 else '' for i in k1])
                G.add_edge(new_k, k1)

    print 'Is empty string or space string present in divergence file? ', str(empty_string)
    print_graph_metrics(G)
    print 'finished. returning graph...'
    return G

def print_graph_metrics(G):
    print 'num nodes: ' + str(len(G.nodes()))
    print 'num edges: ' + str(len(G.edges()))
    print 'printing info about the graph...'
    print nx.info(G)
    print 'is the graph connected? ',
    print nx.is_connected(G)
    print 'number of connected component in the graph: ',
    print nx.algorithms.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)
    print 'number of nodes in largest connected component: ',
    print len(largest_cc)
    # print nx.algorithms.diameter(largest_cc)
    print 'density of graph: ',
    print nx.density(G)
    if nx.is_connected(G):
        print 'Graph is connected. Diameter: ',str(nx.diameter(G)),



def build_undirected_global_graph_from_divergence_files(inputs = [path+'crawlDivergenceJPL.jl',
            path+'crawlDivergenceHG.jl', path+'crawlDivergenceNYU.jl'], ignore_empty_string=True,ascii_only=True):
    G = nx.Graph()
    empty_string = False
    for input in inputs:
        obj = json.load(codecs.open(input, 'r', 'utf-8'))

        for k, v in obj.items():
            # print k, v
            new_k = k
            if k == '' or k == ' ':
                empty_string = True
                print obj[k]
                if ignore_empty_string:
                    continue
                else:
                    new_k = 'empty_string'
            if ascii_only:
                new_k = ''.join([i if ord(i) < 128 else '' for i in new_k])
            for k1, v1, in v.items():

                if k1 == '' or k1 == ' ':
                    empty_string = True

                    if ignore_empty_string:
                        continue
                    else:
                        G.add_edge(new_k, 'empty_string')
                else:
                    if ascii_only:
                        k1 = ''.join([i if ord(i) < 128 else '' for i in k1])
                    G.add_edge(new_k, k1)
    print 'Is empty string or space string present in divergence file? ', str(empty_string)
    # print_graph_metrics(G)
    print 'finished. returning graph.'
    return G


def connected_components_analysis(input = path+'crawlDivergenceJPL.jl', G=None, global_graph=True, ignore_empty_string=True):
    if G is None:
        if global_graph is True: # input should be a list of file names
            G=build_undirected_global_graph_from_divergence_files(ignore_empty_string=ignore_empty_string)
        else: # input should be a string (file name)
            G = build_undirected_graph_from_divergence_file(input=input, ignore_empty_string=ignore_empty_string)
    # input and global graph will both be ignored if G is specified as not None
    cc = sorted(nx.connected_components(G), key=len, reverse=True)
    print '\n\n'
    print 'finished generating connected components. Printing connected component info:'
    print '\n'
    H = G.subgraph(cc[0])
    print_graph_metrics(H)

def compute_node_intersection_metrics(inputs = [path+'crawlDivergenceJPL.jl',
            path+'crawlDivergenceHG.jl', path+'crawlDivergenceNYU.jl'], ignore_empty_string=True):
    nyu_nodes = set(build_undirected_graph_from_divergence_file(inputs[2]).nodes())
    hg_nodes = set(build_undirected_graph_from_divergence_file(inputs[1]).nodes())
    jpl_nodes = set(build_undirected_graph_from_divergence_file(inputs[0]).nodes())
    global_nodes = set(build_undirected_global_graph_from_divergence_files(inputs=inputs).nodes())
    print 'jaccard metrics for nyu-hg'
    print_jaccard_metrics(nyu_nodes, hg_nodes)
    print 'jaccard metrics for hg-jpl'
    print_jaccard_metrics(hg_nodes, jpl_nodes)
    print 'jaccard metrics for nyu-jpl'
    print_jaccard_metrics(nyu_nodes, jpl_nodes)
    # print 'jaccard metrics for nyu-global'
    # print_jaccard_metrics(nyu_nodes, global_nodes)
    # print 'jaccard metrics for hg-global'
    # print_jaccard_metrics(hg_nodes, global_nodes)
    # print 'jaccard metrics for jpl-global'
    # print_jaccard_metrics(global_nodes, jpl_nodes)
    # print 'jaccard metrics for global-global'
    # print_jaccard_metrics(global_nodes, global_nodes)

def print_jaccard_metrics(set1, set2):
    print 'size of intersection, union, jaccard: ',str(len(set1.intersection(set2))),',',\
        str(len(set1.union(set2))),',',str(len(set1.intersection(set2))*1.0/len(set1.union(set2)))
    # print 'size of union: ',
    # print 'jaccard: ',

def get_touched_nodes_single_graph(input = path+'crawlDivergenceNYU.jl', ignore_empty_string=True, ascii_only=False):
    obj = json.load(codecs.open(input, 'r'))
    empty_string = False
    G = set()
    for k, v in obj.items():
        # print k, v
        new_k = k
        if k == '' or k == ' ':
            empty_string = True
            print obj[k]
            if ignore_empty_string:
                continue
            else:
                new_k = 'empty_string'
        if ascii_only:
            new_k = ''.join([i if ord(i) < 128 else ' ' for i in new_k])
        G.add(new_k)
    print 'Is empty string or space string present in divergence file? ', str(empty_string)
    # print_graph_metrics(G)
    print 'finished. returning touched nodes set...'
    return G

def colored_nodes_global_graphs(inputs = [path+'crawlDivergenceJPL.jl',
            path+'crawlDivergenceHG.jl', path+'crawlDivergenceNYU.jl'], touched_nodes=path+'crawlDivergenceHG.jl',
                                output_dot=path+'HG-colored-global.dot',
                                ignore_empty_string=True, touched_nodes_neighbors_only=False):
    """
    We make an exception for u'%URL%' which is causing tremendous problems. We replace it with URL
    :param inputs:
    :param touched_nodes:
    :param output_dot:
    :param ignore_empty_string:
    :return:
    """
    global_graph = build_undirected_global_graph_from_divergence_files(inputs=inputs,
                                                                ignore_empty_string=ignore_empty_string)
    # global_graph = build_undirected_graph_from_divergence_file(input=touched_nodes, ignore_empty_string=ignore_empty_string)
    touched_nodes_set = set()
    if touched_nodes is None:
        nyu_touched_nodes = get_touched_nodes_single_graph(inputs[2],ignore_empty_string=ignore_empty_string)
        jpl_touched_nodes = get_touched_nodes_single_graph(inputs[0], ignore_empty_string=ignore_empty_string)
        hg_touched_nodes = get_touched_nodes_single_graph(inputs[1], ignore_empty_string=ignore_empty_string)
        touched_nodes_set = nyu_touched_nodes.union(jpl_touched_nodes).union(hg_touched_nodes)
    else:
        touched_nodes_set = get_touched_nodes_single_graph(touched_nodes,ignore_empty_string=ignore_empty_string)


    forbidden = set()
    G = pg.AGraph(strict=False, directed=False)
    # if touched_nodes_neighbors_only is False:
    #     for n in global_graph.nodes():
    #         if n == u'%URL%':
    #             n = 'URL'
    #         try:
    #             G.add_node(n)
    #         except TypeError as e:
    #             print e
    #             print n
    #             forbidden.add(n)

            # print n.encode('ascii', 'ignore')
    allowed_nodes = set()
    for edge in global_graph.edges():


        if (edge[0] in touched_nodes_set and edge[1] in touched_nodes_set) or touched_nodes_neighbors_only is False: # change and
            #to or if you want this to be non-strict (make sure to modify the output file as well)
            if edge[0] == u'%URL%':
                    new_edge = list(edge)
                    new_edge[0] = 'URL'
                    edge = tuple(new_edge)
            if edge[1] == u'%URL%':
                    new_edge = list(edge)
                    new_edge[1] = 'URL'
                    edge = tuple(new_edge)
        else:
                continue
        try:
            G.add_node(edge[0])
            G.add_node(edge[1])
            allowed_nodes.add(edge[1])
            allowed_nodes.add(edge[0])
            G.add_edge(edge)
        except TypeError as e:
            print e
            print edge

    # for n in allowed_nodes:
    #     G.add_node(n)


    print 'total number of nodes...',str(len(global_graph.nodes()))
    print 'touched nodes...', str(len(touched_nodes_set))
    print 'number of nodes in graph...',str(len(G.nodes()))

    print 'allowed nodes length...',str(len(allowed_nodes))
    # touched_nodes_att = dict()
    for s in set(global_graph.nodes()):

        if s in forbidden:
            print 'forbidden node: ',s
            continue
        elif s == u'%URL%':
            s = 'URL'

        if s not in allowed_nodes:
            continue
        k = G.get_node(s)
        if s in touched_nodes_set:

            k.attr['fillcolor'] = 'gray'

        # else:
        #     k.attr['fillcolor'] = 'gray100'
    print u'%URL%' in forbidden
    # nx.set_node_attributes(global_graph, 'fillcolor', touched_nodes_att)
    # nx_agraph.write_dot(global_graph, output_dot)
    # nx.write_edgelist(global_graph,output_dot,data=True)
    G.write(output_dot)


def degree_distributions(inputs = [path+'crawlDivergenceJPL.jl',
            path+'crawlDivergenceHG.jl', path+'crawlDivergenceNYU.jl'], touched_nodes=None,#path+'crawlDivergenceHG.jl',
                                ignore_empty_string=True):
    global_graph = build_undirected_global_graph_from_divergence_files(inputs=inputs,
                                                                       ignore_empty_string=ignore_empty_string)
    # global_graph = build_undirected_graph_from_divergence_file(input=touched_nodes, ignore_empty_string=ignore_empty_string)
    touched_nodes_set = set()
    if touched_nodes is None:
        nyu_touched_nodes = get_touched_nodes_single_graph(inputs[2], ignore_empty_string=ignore_empty_string)
        jpl_touched_nodes = get_touched_nodes_single_graph(inputs[0], ignore_empty_string=ignore_empty_string)
        hg_touched_nodes = get_touched_nodes_single_graph(inputs[1], ignore_empty_string=ignore_empty_string)
        touched_nodes_set = nyu_touched_nodes.union(jpl_touched_nodes).union(hg_touched_nodes)
    else:
        touched_nodes_set = get_touched_nodes_single_graph(touched_nodes, ignore_empty_string=ignore_empty_string)

    touched_nodes_degrees = list()
    for t in touched_nodes_set:
        # touched_nodes_degrees[t] = global_graph.degree(t)
        if type(global_graph.degree(t)) is dict: # I'm not sure why this is happening but it is.
            touched_nodes_degrees += global_graph.degree(t).values()
        else:
            touched_nodes_degrees.append(global_graph.degree(t))

    degree_hist = touched_nodes_degrees
    degree_hist.sort()
    # print degree_hist
    log_k = list()
    log_freq = list()
    total_sum = np.sum(degree_hist)
    print 'avg. degree is : ',str(np.mean(degree_hist))
    print 'std. dev. of degree is : ', str(np.std(degree_hist))
    k_vec = list()
    freq_vec = list()
    print 'total sum of degree hist. list is...', str(total_sum)
    # degree_dict = dict()
    if degree_hist[0] > 0:
        log_k.append(0)
        k_vec.append(0)
        log_freq.append(math.log(degree_hist[0] * 1.0 / total_sum))
        freq_vec.append(degree_hist[0] * 1.0 / total_sum)
    for i in range(1, len(degree_hist)):
        if degree_hist[i] == 0:
            continue
        else:
            k_vec.append(i)
            freq_vec.append(degree_hist[i] * 1.0 / total_sum)
            log_k.append(math.log(i))
            log_freq.append(math.log(degree_hist[i] * 1.0 / total_sum))
    # plt.loglog(degree_sequence, 'b-', marker='o')
    regr = linear_model.LinearRegression()
    print len(log_k)
    print len(log_freq)
    # Train the model using the training sets
    regr.fit(np.array(log_k).reshape(-1, 1), np.array(log_freq).reshape(-1, 1))

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    plt.loglog(k_vec, freq_vec, 'ro')

    plt.title("Degree distribution plot")
    plt.ylabel("Prob(k)")
    plt.xlabel("k")
    plt.show()



# colored_nodes_global_graphs()
# connected_components_analysis()
# degree_distributions()
build_undirected_graph_from_divergence_file()
print len(get_touched_nodes_single_graph())