import networkx as nx
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)

def build_bipartite_graph(jobs_df: pd.DataFrame) -> nx.Graph:
    """
    Build a bipartite graph connecting jobs to skills.
    
    Args:
        jobs_df: DataFrame with columns 'job_id' and 'skills_detected'
    
    Returns:
        NetworkX bipartite graph
    """
    G = nx.Graph()
    
    for _, row in jobs_df.iterrows():
        job_id = row.get("job_id", f"job_{_}")
        skills = row.get("skills_detected", [])
        
        # Add job node
        G.add_node(job_id, bipartite=0, node_type="job")
        
        # Add skill nodes and edges
        for skill in skills:
            if skill:  # Skip empty skills
                G.add_node(skill, bipartite=1, node_type="skill")
                G.add_edge(job_id, skill)
    
    return G

def build_skill_cooccurrence_graph(jobs_df: pd.DataFrame) -> nx.Graph:
    """
    Build a skill co-occurrence graph (projection of bipartite graph).
    Two skills are connected if they appear together in at least one job.
    Edge weight = number of jobs where both skills appear.
    
    Args:
        jobs_df: DataFrame with 'skills_detected' column
    
    Returns:
        NetworkX graph of skill co-occurrences
    """
    G = nx.Graph()
    
    # Count co-occurrences
    cooccurrence = Counter()
    
    for _, row in jobs_df.iterrows():
        skills = row.get("skills_detected", [])
        skills = [s for s in skills if s]  # Filter empty
        
        # Add all pairs of skills that co-occur
        for i, skill1 in enumerate(skills):
            G.add_node(skill1)
            for skill2 in skills[i+1:]:
                G.add_node(skill2)
                pair = tuple(sorted([skill1, skill2]))
                cooccurrence[pair] += 1
    
    # Add edges with weights
    for (skill1, skill2), weight in cooccurrence.items():
        G.add_edge(skill1, skill2, weight=weight, cooccurrence_count=weight)
    
    return G

def compute_centralities(graph: nx.Graph) -> pd.DataFrame:
    """
    Compute various centrality measures for nodes in the graph.
    
    Returns:
        DataFrame with columns: node, degree, betweenness, closeness, eigenvector
    """
    if len(graph.nodes()) == 0:
        return pd.DataFrame(columns=["node", "degree", "betweenness", "closeness", "eigenvector"])
    
    centralities = {
        "node": list(graph.nodes()),
        "degree": nx.degree_centrality(graph),
        "betweenness": nx.betweenness_centrality(graph),
        "closeness": nx.closeness_centrality(graph),
    }
    
    # Eigenvector centrality (can fail for disconnected graphs)
    try:
        centralities["eigenvector"] = nx.eigenvector_centrality(graph, max_iter=1000)
    except:
        centralities["eigenvector"] = {node: 0.0 for node in graph.nodes()}
    
    # Convert to DataFrame
    df = pd.DataFrame({
        "node": centralities["node"],
        "degree": [centralities["degree"].get(n, 0) for n in centralities["node"]],
        "betweenness": [centralities["betweenness"].get(n, 0) for n in centralities["node"]],
        "closeness": [centralities["closeness"].get(n, 0) for n in centralities["node"]],
        "eigenvector": [centralities["eigenvector"].get(n, 0) for n in centralities["node"]],
    })
    
    # Add weighted degree if graph has edge weights
    if any("weight" in graph[u][v] for u, v in graph.edges()):
        weighted_degree = {}
        for node in graph.nodes():
            weighted_degree[node] = sum(
                graph[node][neighbor].get("weight", 1)
                for neighbor in graph.neighbors(node)
            )
        df["weighted_degree"] = [weighted_degree.get(n, 0) for n in df["node"]]
    
    return df.sort_values("degree", ascending=False)

def detect_communities(graph: nx.Graph, algorithm: str = "louvain") -> Dict[str, int]:
    """
    Detect communities in the graph using community detection algorithms.
    
    Args:
        graph: NetworkX graph
        algorithm: 'louvain' or 'greedy_modularity'
    
    Returns:
        Dictionary mapping node -> community_id
    """
    if len(graph.nodes()) == 0:
        return {}
    
    try:
        if algorithm == "louvain":
            import community.community_louvain as community_louvain
            communities = community_louvain.best_partition(graph)
        elif algorithm == "greedy_modularity":
            from networkx.algorithms import community
            communities_generator = community.greedy_modularity_communities(graph)
            communities = {}
            for i, comm in enumerate(communities_generator):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return communities
    except ImportError:
        logger.warning("python-louvain not installed, using greedy_modularity")
        from networkx.algorithms import community
        communities_generator = community.greedy_modularity_communities(graph)
        communities = {}
        for i, comm in enumerate(communities_generator):
            for node in comm:
                communities[node] = i
        return communities
    except Exception as e:
        logger.error(f"Error detecting communities: {e}")
        # Return trivial communities (each node is its own community)
        return {node: i for i, node in enumerate(graph.nodes())}

def get_skill_importance_scores(jobs_df: pd.DataFrame, 
                                user_skills: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute comprehensive importance scores for skills combining:
    - Frequency (how often it appears)
    - Centrality (position in skill network)
    - Co-occurrence (connections to other important skills)
    
    Args:
        jobs_df: DataFrame with jobs and skills
        user_skills: Optional list of user skills for personalized scoring
    
    Returns:
        DataFrame with skill importance metrics
    """
    # Build co-occurrence graph
    G = build_skill_cooccurrence_graph(jobs_df)
    
    if len(G.nodes()) == 0:
        return pd.DataFrame(columns=["skill", "frequency", "degree_centrality", 
                                    "betweenness_centrality", "importance_score"])
    
    # Frequency
    all_skills = []
    for _, row in jobs_df.iterrows():
        all_skills.extend(row.get("skills_detected", []))
    freq = Counter(all_skills)
    total_jobs = len(jobs_df)
    
    # Centralities
    centrality_df = compute_centralities(G)
    
    # Merge frequency and centrality
    importance_df = pd.DataFrame({
        "skill": list(freq.keys()),
        "frequency": [freq[skill] for skill in freq.keys()],
        "frequency_pct": [freq[skill] / total_jobs * 100 if total_jobs > 0 else 0 
                          for skill in freq.keys()],
    })
    
    # Merge with centralities
    importance_df = importance_df.merge(
        centrality_df[["node", "degree", "betweenness", "closeness", "eigenvector"]],
        left_on="skill",
        right_on="node",
        how="left"
    ).drop(columns=["node"])
    
    # Fill NaN with 0 for skills not in graph
    importance_df = importance_df.fillna(0)
    
    # Compute composite importance score
    # Normalize each metric to 0-1 range
    max_freq = importance_df["frequency"].max() if importance_df["frequency"].max() > 0 else 1
    max_degree = importance_df["degree"].max() if importance_df["degree"].max() > 0 else 1
    max_betweenness = importance_df["betweenness"].max() if importance_df["betweenness"].max() > 0 else 1
    
    importance_df["normalized_frequency"] = importance_df["frequency"] / max_freq
    importance_df["normalized_degree"] = importance_df["degree"] / max_degree
    importance_df["normalized_betweenness"] = importance_df["betweenness"] / max_betweenness
    
    # Weighted combination (can be tuned)
    importance_df["importance_score"] = (
        0.4 * importance_df["normalized_frequency"] +
        0.4 * importance_df["normalized_degree"] +
        0.2 * importance_df["normalized_betweenness"]
    )
    
    # Boost score if user already has this skill (for recommendations)
    if user_skills:
        user_skills_set = set(user_skills)
        importance_df["user_has"] = importance_df["skill"].isin(user_skills_set)
    else:
        importance_df["user_has"] = False
    
    return importance_df.sort_values("importance_score", ascending=False)

def find_bridge_skills(graph: nx.Graph, top_n: int = 10) -> List[str]:
    """
    Find skills that act as bridges between communities (high betweenness).
    
    Returns:
        List of top bridge skills
    """
    if len(graph.nodes()) == 0:
        return []
    
    betweenness = nx.betweenness_centrality(graph)
    sorted_skills = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in sorted_skills[:top_n]]

def get_skill_paths(graph: nx.Graph, skill1: str, skill2: str) -> List[List[str]]:
    """
    Find all shortest paths between two skills.
    Useful for understanding skill relationships.
    """
    if skill1 not in graph or skill2 not in graph:
        return []
    
    try:
        paths = list(nx.all_shortest_paths(graph, skill1, skill2))
        return paths
    except nx.NetworkXNoPath:
        return []

