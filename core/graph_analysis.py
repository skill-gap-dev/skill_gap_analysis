"""
Graph analysis module for skill gap analysis.
Implements bipartite graphs, skill co-occurrence networks, centrality measures,
and community detection.
"""
import networkx as nx
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def build_skill_cooccurrence_graph(jobs_df: pd.DataFrame) -> nx.Graph:
    """
    Build a skill co-occurrence graph (projection of bipartite graph).
    Two skills are connected if they appear together in at least one job.
    Edge weight = number of jobs where both skills co-occur.
    
    Args:
        jobs_df: DataFrame with columns 'job_id' and 'skills_detected'
        
    Returns:
        NetworkX weighted graph of skills
    """
    G = nx.Graph()
    
    # Count co-occurrences
    cooccurrence = Counter()
    
    for _, row in jobs_df.iterrows():
        skills = row.get("skills_detected", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        elif not isinstance(skills, list):
            skills = []
        
        # Remove duplicates and filter empty
        skills = list(set([s for s in skills if s]))
        
        # Count pairs
        for i, skill1 in enumerate(skills):
            for skill2 in skills[i+1:]:
                pair = tuple(sorted([skill1, skill2]))
                cooccurrence[pair] += 1
    
    # Add edges with weights
    for (skill1, skill2), weight in cooccurrence.items():
        G.add_edge(skill1, skill2, weight=weight)
    
    return G


def compute_centralities(graph: nx.Graph) -> pd.DataFrame:
    """
    Compute various centrality measures for nodes in the graph.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        DataFrame with columns: node, degree, betweenness, closeness, eigenvector
    """
    if len(graph.nodes()) == 0:
        return pd.DataFrame(columns=["node", "degree", "betweenness", "closeness", "eigenvector"])
    
    # Degree centrality
    degree_cent = nx.degree_centrality(graph)
    
    # Check if graph has weighted edges
    has_weights = False
    if graph.number_of_edges() > 0:
        # Check if any edge has a weight attribute
        sample_edge = list(graph.edges(data=True))[0]
        has_weights = "weight" in sample_edge[2] if len(sample_edge) > 2 else False
    
    # Betweenness centrality (only if graph has edges)
    if graph.number_of_edges() > 0:
        try:
            betweenness_cent = nx.betweenness_centrality(graph, weight="weight" if has_weights else None)
        except:
            betweenness_cent = nx.betweenness_centrality(graph)
    else:
        betweenness_cent = {node: 0.0 for node in graph.nodes()}
    
    # Closeness centrality (only for connected graphs)
    try:
        closeness_cent = nx.closeness_centrality(graph, distance="weight" if has_weights else None)
    except:
        try:
            closeness_cent = nx.closeness_centrality(graph)
        except:
            closeness_cent = {node: 0.0 for node in graph.nodes()}
    
    # Eigenvector centrality
    try:
        eigenvector_cent = nx.eigenvector_centrality(graph, weight="weight" if has_weights else None, max_iter=1000)
    except:
        try:
            eigenvector_cent = nx.eigenvector_centrality(graph, max_iter=1000)
        except:
            eigenvector_cent = {node: 0.0 for node in graph.nodes()}
    
    # Weighted degree (sum of edge weights)
    if has_weights:
        weighted_degree = dict(graph.degree(weight="weight"))
    else:
        weighted_degree = dict(graph.degree())
    
    # Combine into DataFrame
    nodes = list(graph.nodes())
    data = {
        "node": nodes,
        "degree": [degree_cent.get(n, 0) for n in nodes],
        "betweenness": [betweenness_cent.get(n, 0) for n in nodes],
        "closeness": [closeness_cent.get(n, 0) for n in nodes],
        "eigenvector": [eigenvector_cent.get(n, 0) for n in nodes],
        "weighted_degree": [weighted_degree.get(n, 0) for n in nodes],
    }
    
    return pd.DataFrame(data).sort_values("degree", ascending=False)


def detect_communities(graph: nx.Graph, algorithm: str = "louvain", resolution: float = 1.0, 
                       min_community_size: int = 2) -> Dict[str, int]:
    """
    Detect communities in the graph using improved community detection algorithms.
    
    Args:
        graph: NetworkX graph
        algorithm: 'louvain', 'greedy_modularity', 'label_propagation', or 'best'
        resolution: Resolution parameter for Louvain (higher = more communities, default 1.0)
        min_community_size: Minimum size for a community (smaller communities will be merged)
        
    Returns:
        Dictionary mapping node -> community_id
    """
    if len(graph.nodes()) == 0:
        return {}
    
    if graph.number_of_edges() == 0:
        return {node: 0 for node in graph.nodes()}
    
    try:
        has_weights = any("weight" in graph[u][v] for u, v in graph.edges())
        weight_param = "weight" if has_weights else None
        
        if algorithm == "best":
            best_communities = None
            best_modularity = -1
            best_algorithm = None
            
            for algo in ["louvain", "greedy_modularity", "label_propagation"]:
                try:
                    comms = detect_communities(graph, algo, resolution, min_community_size)
                    if comms:
                        mod = compute_modularity(graph, comms, weight_param)
                        if mod > best_modularity:
                            best_modularity = mod
                            best_communities = comms
                            best_algorithm = algo
                except:
                    continue
            
            if best_communities:
                logger.info(f"Best algorithm: {best_algorithm} (modularity: {best_modularity:.3f})")
                return best_communities
        
        if algorithm == "louvain":
            try:
                import community.community_louvain as community_louvain
                communities = community_louvain.best_partition(
                    graph, 
                    weight=weight_param,
                    resolution=resolution
                )
            except ImportError:
                from networkx.algorithms import community
                communities_generator = community.greedy_modularity_communities(
                    graph, weight=weight_param
                )
                communities = {}
                for i, comm in enumerate(communities_generator):
                    for node in comm:
                        communities[node] = i
        elif algorithm == "greedy_modularity":
            from networkx.algorithms import community
            communities_generator = community.greedy_modularity_communities(
                graph, weight=weight_param
            )
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
        elif algorithm == "label_propagation":
            from networkx.algorithms import community
            communities_generator = community.asyn_lpa_communities(graph, weight=weight_param)
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'louvain', 'greedy_modularity', 'label_propagation', or 'best'")
        
        if min_community_size > 1:
            communities = merge_small_communities(graph, communities, min_community_size)
        
        return communities
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        return {node: i for i, node in enumerate(graph.nodes())}


def compute_modularity(graph: nx.Graph, communities: Dict[str, int], weight: Optional[str] = None) -> float:
    """
    Compute modularity score for a community partition.
    
    Args:
        graph: NetworkX graph
        communities: Dictionary mapping node -> community_id
        weight: Edge weight attribute name
        
    Returns:
        Modularity score (higher is better)
    """
    try:
        from networkx.algorithms import community
        partition = []
        comm_dict = {}
        for node, comm_id in communities.items():
            if comm_id not in comm_dict:
                comm_dict[comm_id] = []
            comm_dict[comm_id].append(node)
        partition = list(comm_dict.values())
        return community.modularity(graph, partition, weight=weight)
    except:
        return 0.0


def merge_small_communities(graph: nx.Graph, communities: Dict[str, int], 
                           min_size: int = 2) -> Dict[str, int]:
    """
    Merge communities smaller than min_size into the nearest larger community.
    
    Args:
        graph: NetworkX graph
        communities: Dictionary mapping node -> community_id
        min_size: Minimum community size
        
    Returns:
        Updated communities dictionary
    """
    comm_sizes = Counter(communities.values())
    small_communities = {comm_id for comm_id, size in comm_sizes.items() if size < min_size}
    
    if not small_communities:
        return communities
    
    updated_communities = communities.copy()
    
    for node, comm_id in communities.items():
        if comm_id in small_communities:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_comm_ids = [communities.get(n, comm_id) for n in neighbors 
                                    if communities.get(n, comm_id) not in small_communities]
                if neighbor_comm_ids:
                    most_common_comm = Counter(neighbor_comm_ids).most_common(1)[0][0]
                    updated_communities[node] = most_common_comm
                else:
                    largest_comm = comm_sizes.most_common(1)[0][0]
                    updated_communities[node] = largest_comm
    
    return updated_communities

def find_bridge_skills(centralities_df: pd.DataFrame, top_n: int = 10) -> List[str]:
    """
    Find top bridge skills by betweenness centrality.
    
    Args:
        centralities_df: DataFrame with centrality measures
        top_n: Number of skills to return
        
    Returns:
        List of skill names
    """
    betweenness = dict(zip(centralities_df["node"], centralities_df["betweenness"]))
    
    sorted_skills = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills[:top_n]]


def plot_skill_network(
    graph: nx.Graph,
    communities: Dict[str, int],
    highlight_skills: List[str] = None,
    user_skills: List[str] = None,
):
    """
    Create an interactive network visualization using pyvis.
    
    Args:
        graph: NetworkX skill co-occurrence graph
        communities: Dictionary mapping skill -> community_id
        highlight_skills: List of skills to highlight (e.g., bridge skills)
        user_skills: List of user skills to highlight differently in the graph
        
    Returns:
        pyvis Network object or None
    """
    if len(graph.nodes()) == 0 or graph.number_of_edges() == 0:
        return None
    
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis not installed. Install with: pip install pyvis")
        return None
    
    try:
        net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=False)
        
        has_weights = any("weight" in graph[u][v] for u, v in graph.edges())
        
        highlight_skills_set = set(highlight_skills) if highlight_skills else set()
        user_skills_set = set(user_skills) if user_skills else set()
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#ecf0f1", "#34495e", "#e67e22", "#16a085"]
        
        pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)
        
        for node_id in graph.nodes():
            x, y = pos[node_id]
            
            try:
                degree = graph.degree(node_id, weight='weight') if has_weights else graph.degree(node_id)
            except:
                degree = graph.degree(node_id)
            
            neighbors = list(graph.neighbors(node_id))
            comm_id = communities.get(node_id, 0) if communities else 0
            
            is_bridge = node_id in highlight_skills_set
            is_user_skill = node_id in user_skills_set

            # Color by community (do not change color/shape for user skills to preserve community identity)
            node_color = colors[comm_id % len(colors)] if communities else "#3498db"
            
            # Slightly bigger size for bridge skills; user skills keep same size to avoid confusion in communities
            node_size = 40 if is_bridge else max(20, min(30, int(15 + degree * 2)))
            
            title = f"{node_id}"
            title += f" - Degree: {degree}"
            title += f" - Connections: {len(neighbors)}"
            if communities:
                title += f" - Community: {comm_id}"
            if has_weights:
                total_weight = sum(graph[node_id][n].get('weight', 1) for n in neighbors)
                title += f" - Total Weight: {total_weight:.1f}"
            if is_bridge:
                title += " - ⭐ Bridge Skill"
            if is_user_skill:
                title += " - ⭐ User Skill"
            
            net.add_node(
                node_id,
                label=(
                    f"⭐ {node_id[:13]}..." if len(node_id) > 15 else f"⭐ {node_id}"
                )
                if is_user_skill
                else (node_id[:15] + "..." if len(node_id) > 15 else node_id),
                title=title,
                x=x * 1000,
                y=y * 1000,
                size=node_size,
                color=node_color,
                borderWidth=4 if is_bridge else 2,
                borderColor="#ffffff",
                font={"size": 16, "face": "Arial", "color": "white"},
                shape="dot",
            )
        
        edge_weights_list = []
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1) if has_weights else 1
            edge_weights_list.append(weight)
        
        if edge_weights_list:
            min_weight = min(edge_weights_list)
            max_weight = max(edge_weights_list)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1
        else:
            min_weight = max_weight = 1
            weight_range = 1
        
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1) if has_weights else 1
            
            edge_width = max(1, min(8, 1 + (weight - min_weight) / weight_range * 7)) if weight_range > 0 else 2
            
            edge_title = f"{u} ↔  {v} Co-occurrences: {int(weight)}"
            
            net.add_edge(
                u, v,
                width=edge_width,
                title=edge_title,
                color={"color": "rgba(136, 136, 136, 0.6)", "highlight": "#b8e994"},
                smooth={"type": "continuous", "roundness": 0.5}
            )
        
        net.set_options("""
        {
          "physics": {
            "enabled": false,
            "stabilization": {"enabled": false}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": false,
            "hideNodesOnDrag": false
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "Arial",
              "color": "white"
            },
            "scaling": {
              "min": 10,
              "max": 40
            }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "roundness": 0.5
            },
            "shadow": {
              "enabled": false
            }
          }
        }
        """)
        
        return net
        
    except Exception as e:
        logger.error(f"Error creating network visualization: {e}")
        return None


def get_skill_importance_scores(G, centrality_df, jobs_df: pd.DataFrame, user_skills: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute comprehensive importance scores for skills combining:
    - Frequency (how often it appears)
    - Centrality (position in skill network)
    - Co-occurrence (connections to other important skills)
    
    Args:
        G: Skill co-occurrence graph
        centrality_df: DataFrame with centrality measures
        jobs_df: DataFrame with jobs and skills_detected column
        user_skills: Optional list of user skills for personalized scoring
        
    Returns:
        DataFrame with skill importance metrics
    """
    
    if len(G.nodes()) == 0:
        return pd.DataFrame(columns=["skill", "frequency", "degree_centrality", 
                                    "betweenness_centrality", "importance_score"])
    
    all_skills = []
    for _, row in jobs_df.iterrows():
        skills = row.get("skills_detected", [])
        if isinstance(skills, list):
            all_skills.extend(skills)
        elif isinstance(skills, str):
            all_skills.extend([s.strip() for s in skills.split(",") if s.strip()])
    
    freq = Counter(all_skills)
    total_jobs = len(jobs_df)
    
    importance_df = pd.DataFrame({
        "skill": list(freq.keys()),
        "frequency": [freq[skill] for skill in freq.keys()],
        "frequency_pct": [freq[skill] / total_jobs * 100 if total_jobs > 0 else 0 
                          for skill in freq.keys()],
    })
    
    importance_df = importance_df.merge(
        centrality_df[["node", "degree", "betweenness", "closeness", "eigenvector"]],
        left_on="skill",
        right_on="node",
        how="left"
    ).drop(columns=["node"])
    
    importance_df = importance_df.fillna(0)
    
    max_freq = importance_df["frequency"].max() if importance_df["frequency"].max() > 0 else 1
    max_degree = importance_df["degree"].max() if importance_df["degree"].max() > 0 else 1
    max_betweenness = importance_df["betweenness"].max() if importance_df["betweenness"].max() > 0 else 1
    
    importance_df["normalized_frequency"] = importance_df["frequency"] / max_freq
    importance_df["normalized_degree"] = importance_df["degree"] / max_degree
    importance_df["normalized_betweenness"] = importance_df["betweenness"] / max_betweenness
    
    importance_df["importance_score"] = (
        0.4 * importance_df["normalized_frequency"] +
        0.4 * importance_df["normalized_degree"] +
        0.2 * importance_df["normalized_betweenness"]
    )
    
    if user_skills:
        user_skills_set = set(user_skills)
        importance_df["user_has"] = importance_df["skill"].isin(user_skills_set)
    else:
        importance_df["user_has"] = False
    
    return importance_df.sort_values("importance_score", ascending=False)