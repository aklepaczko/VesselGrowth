import networkx as nx

from akle.cco.vessel import Vessel


def _add_vessel(graph: nx.Graph, vessel: Vessel):
    graph.add_node(vessel, label=vessel.index)
    if vessel.is_parent:
        _add_vessel(graph, vessel.son)
        graph.add_edge(vessel, vessel.son)
        _add_vessel(graph, vessel.daughter)
        graph.add_edge(vessel, vessel.daughter)


def create_vasculature_graph(vasculature: dict[str, Vessel | list[Vessel]]) -> nx.Graph:
    root = vasculature['root']
    out_graph = nx.Graph()
    _add_vessel(out_graph, root)
    return out_graph
