# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 19:19:56 2025

@author: xzhou
"""

#!/usr/bin/env python3
"""
Phase-Augmented Dynamic Traffic Assignment (PA-DTA) Implementation
NumPy-only version without PyTorch tensors

This implementation uses only NumPy and standard Python libraries
for maximum compatibility and simplicity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import networkx as nx
import time
import csv
import json
from collections import defaultdict
import math


@dataclass
class NetworkParams:
    """Network configuration parameters"""
    time_horizon: int = 120  # 2 hours in minutes
    delta_t: float = 1.0     # 1-minute time steps
    num_phases: int = 4      # Number of traffic phases
    demand_multiplier: float = 1.0  # Scale factor for demand


class NetworkDataLoader:
    """Load and process real network data using only standard libraries"""
    
    def __init__(self):
        self.nodes_df = None
        self.links_df = None
        self.routes_df = None
        self.network_graph = None
        
    def load_data(self, node_file='node.csv', link_file='link.csv', route_file='route_assignment.csv'):
        """Load network data from CSV files"""
        try:
            print("Loading network data...")
            
            # Load nodes
            self.nodes_df = pd.read_csv(node_file)
            print(f"Loaded {len(self.nodes_df)} nodes")
            
            # Load links
            self.links_df = pd.read_csv(link_file)
            print(f"Loaded {len(self.links_df)} links")
            
            # Load routes (may be empty)
            try:
                self.routes_df = pd.read_csv(route_file)
                print(f"Loaded {len(self.routes_df)} route records")
            except:
                print("Route assignment file empty or not found - will generate synthetic routes")
                self.routes_df = pd.DataFrame()
            
            # Create network graph
            self._create_network_graph()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _create_network_graph(self):
        """Create NetworkX graph from data"""
        self.network_graph = nx.DiGraph()
        
        # Add nodes
        for _, node in self.nodes_df.iterrows():
            self.network_graph.add_node(
                node['node_id'],
                x=node['x_coord'],
                y=node['y_coord'],
                zone_id=node.get('zone_id', 0)
            )
        
        # Add links as edges
        for _, link in self.links_df.iterrows():
            self.network_graph.add_edge(
                link['from_node_id'],
                link['to_node_id'],
                link_id=link['link_id'],
                length=link['length'],
                capacity=link['capacity'],
                free_speed=link['free_speed'],
                lanes=link['lanes'],
                vdf_alpha=link.get('VDF_alpha', 0.15),
                vdf_beta=link.get('VDF_beta', 4),
                fftt=link.get('VDF_fftt', link['length'] / max(link['free_speed'], 1) * 60),
                base_volume=link.get('base_volume', 0)
            )
    
    def get_network_stats(self):
        """Get basic network statistics"""
        zone_count = len(self.nodes_df[self.nodes_df['zone_id'] > 0]) if 'zone_id' in self.nodes_df.columns else 0
        stats = {
            'num_nodes': len(self.nodes_df),
            'num_links': len(self.links_df),
            'num_zones': zone_count,
            'total_capacity': self.links_df['capacity'].sum(),
            'avg_link_length': self.links_df['length'].mean(),
            'total_network_length': self.links_df['length'].sum()
        }
        return stats


class ModelX:
    """Model X: Queue Dynamics using NumPy arrays"""
    
    def __init__(self, links_df: pd.DataFrame, params: NetworkParams):
        self.links_df = links_df
        self.params = params
        self.A = len(links_df)  # Number of links
        self.T = params.time_horizon
        
        # Create link index mapping
        self.link_id_to_idx = {row['link_id']: idx for idx, row in links_df.iterrows()}
        self.idx_to_link_id = {idx: row['link_id'] for idx, row in links_df.iterrows()}
        
        # Initialize state variables as NumPy arrays
        self.x = np.zeros((self.A, self.T))  # Queue length [veh]
        self.u = np.zeros((self.A, self.T))  # Inflow [veh/min]
        self.g = np.zeros((self.A, self.T))  # Outflow [veh/min]
        
        # Link properties from data
        self.capacity = np.array(links_df['capacity'].values, dtype=float)  # [veh/h]
        
        # Calculate free-flow travel time
        fftt_values = []
        for _, link in links_df.iterrows():
            if 'VDF_fftt' in link and not pd.isna(link['VDF_fftt']):
                fftt_values.append(link['VDF_fftt'])
            else:
                fftt_values.append(link['length'] / max(link['free_speed'], 1) * 60)
        
        self.free_flow_time = np.array(fftt_values, dtype=float)  # [min]
        self.length = np.array(links_df['length'].values, dtype=float)
        
        # VDF parameters
        self.vdf_alpha = np.array([link.get('VDF_alpha', 0.15) for _, link in links_df.iterrows()], dtype=float)
        self.vdf_beta = np.array([link.get('VDF_beta', 4.0) for _, link in links_df.iterrows()], dtype=float)
        
        # Convert hourly capacity to per-minute
        self.capacity_per_min = self.capacity / 60.0
        
        print(f"Model X initialized with {self.A} links, {self.T} time steps")
    
    def update_queue_dynamics(self, inflow: np.ndarray, phase_discharge_rates: np.ndarray) -> Dict:
        """Update queue dynamics with phase-dependent discharge rates"""
        self.u = inflow.copy()
        
        # Reset queues and outflows
        self.x.fill(0.0)
        self.g.fill(0.0)
        
        for t in range(self.T - 1):
            # Calculate phase-weighted discharge capacity
            effective_capacity = np.minimum(
                phase_discharge_rates[:, t], 
                self.capacity_per_min
            )
            
            # Calculate outflow (cannot exceed queue + inflow or capacity)
            available_vehicles = self.x[:, t] + self.u[:, t]
            self.g[:, t] = np.minimum(available_vehicles, effective_capacity)
            
            # Update queue length
            self.x[:, t + 1] = self.x[:, t] + self.u[:, t] - self.g[:, t]
            self.x[:, t + 1] = np.maximum(self.x[:, t + 1], 0.0)  # Ensure non-negative
        
        return {
            'queue': self.x.copy(),
            'inflow': self.u.copy(),
            'outflow': self.g.copy()
        }
    
    def calculate_travel_times(self, current_time: int = None) -> np.ndarray:
        """Calculate link travel times using VDF and queueing delay"""
        if current_time is None:
            # Use average conditions
            avg_volume = np.mean(self.u, axis=1) * 60  # Convert to veh/h
            avg_queue = np.mean(self.x, axis=1)
        else:
            avg_volume = self.u[:, current_time] * 60
            avg_queue = self.x[:, current_time]
        
        # BPR function for congestion delay
        volume_capacity_ratio = avg_volume / np.maximum(self.capacity, 1.0)
        bpr_factor = 1.0 + self.vdf_alpha * np.power(volume_capacity_ratio, self.vdf_beta)
        congestion_time = self.free_flow_time * bpr_factor
        
        # Add queueing delay
        queue_delay = np.zeros_like(avg_queue)
        non_zero_capacity = self.capacity_per_min > 0
        queue_delay[non_zero_capacity] = avg_queue[non_zero_capacity] / self.capacity_per_min[non_zero_capacity]
        
        total_travel_time = congestion_time + queue_delay
        return total_travel_time


class ModelY:
    """Model Y: Phase Control using NumPy arrays"""
    
    def __init__(self, links_df: pd.DataFrame, params: NetworkParams):
        self.links_df = links_df
        self.params = params
        self.A = len(links_df)
        self.P = params.num_phases
        self.T = params.time_horizon
        
        # Phase selection variables
        self.z = np.zeros((self.A, self.P))  # Phase selection probabilities
        
        # Initialize phase parameters
        self._initialize_phase_parameters()
        
        print(f"Model Y initialized with {self.A} links, {self.P} phases")
    
    def _initialize_phase_parameters(self):
        """Initialize phase parameters based on link characteristics"""
        self.phase_params = {}
        
        for idx, (_, link) in enumerate(self.links_df.iterrows()):
            self.phase_params[idx] = {}
            base_capacity = link['capacity'] / 60.0  # Convert to per-minute
            
            for p in range(self.P):
                # Different phases represent different operational conditions
                if p == 0:  # Free flow
                    discharge_rate = base_capacity * 1.0
                    capacity_factor = 1.0
                    congestion_threshold = 0.3
                elif p == 1:  # Light congestion
                    discharge_rate = base_capacity * 0.9
                    capacity_factor = 0.9
                    congestion_threshold = 0.6
                elif p == 2:  # Heavy congestion
                    discharge_rate = base_capacity * 0.7
                    capacity_factor = 0.7
                    congestion_threshold = 0.85
                else:  # Breakdown conditions
                    discharge_rate = base_capacity * 0.5
                    capacity_factor = 0.5
                    congestion_threshold = 1.0
                
                self.phase_params[idx][p] = {
                    'mu': discharge_rate,
                    'phi': capacity_factor,
                    'threshold': congestion_threshold,
                    't0': 0,
                    't3': self.T,
                    'beta': 0.1 + p * 0.05
                }
    
    def select_phases(self, queue_states: np.ndarray, inflow_states: np.ndarray) -> np.ndarray:
        """Select phases based on current traffic conditions"""
        phase_scores = np.zeros((self.A, self.P))
        
        for a in range(self.A):
            # Calculate current demand-to-capacity ratio
            current_inflow = np.mean(inflow_states[a, :]) * 60  # Convert to hourly
            base_capacity = self.links_df.iloc[a]['capacity']
            dcr = current_inflow / max(base_capacity, 1.0)
            
            # Calculate average queue length
            avg_queue = np.mean(queue_states[a, :])
            
            # Score each phase
            for p in range(self.P):
                params = self.phase_params[a][p]
                threshold = params['threshold']
                
                # Higher score for phases that match current conditions
                if dcr <= threshold:
                    score = 1.0 - abs(dcr - threshold * 0.8)
                else:
                    score = max(0.1, 1.0 - (dcr - threshold))
                
                # Adjust score based on queue length
                if avg_queue > 10 and p >= 2:  # Prefer higher-capacity phases when queued
                    score *= 1.2
                elif avg_queue < 2 and p == 0:  # Prefer free-flow phase when uncongested
                    score *= 1.1
                
                phase_scores[a, p] = score
        
        # Softmax to get phase selection probabilities
        self.z = self._softmax(phase_scores)
        return self.z
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def get_discharge_rates(self) -> np.ndarray:
        """Get phase-weighted discharge rates"""
        discharge_rates = np.zeros((self.A, self.T))
        
        for t in range(self.T):
            discharge_rates[:, t] = self._calculate_weighted_discharge_rates()
        
        return discharge_rates
    
    def _calculate_weighted_discharge_rates(self) -> np.ndarray:
        """Calculate weighted average discharge rates based on phase selection"""
        rates = np.zeros(self.A)
        
        for a in range(self.A):
            total_rate = 0.0
            for p in range(self.P):
                weight = self.z[a, p]
                mu = self.phase_params[a][p]['mu']
                total_rate += weight * mu
            rates[a] = total_rate
        
        return rates
    
    def estimate_newell_parameters(self, discrete_states: Dict) -> Dict:
        """Estimate continuous-time Newell parameters from discrete data"""
        newell_params = {}
        
        for a in range(self.A):
            queue_series = discrete_states['queue'][a, :]
            inflow_series = discrete_states['inflow'][a, :]
            outflow_series = discrete_states['outflow'][a, :]
            
            # Find congestion period
            congestion_mask = queue_series > 0.1
            congestion_indices = np.where(congestion_mask)[0]
            
            if len(congestion_indices) > 0:
                t0 = congestion_indices[0]
                t3 = congestion_indices[-1]
                duration = t3 - t0 + 1
                
                # Calculate parameters
                avg_discharge = np.mean(outflow_series[t0:t3+1]) if duration > 0 else 50.0
                peak_inflow = np.max(inflow_series[t0:t3+1]) if duration > 0 else 30.0
                
                # Estimate beta (curvature parameter)
                if duration > 0 and peak_inflow > avg_discharge:
                    beta = 3 * (peak_inflow - avg_discharge)**2 / duration
                else:
                    beta = 0.1
                
                newell_params[a] = {
                    'T0': t0, 'T3': t3, 'P': duration,
                    'mu': avg_discharge, 'lambda': peak_inflow, 'beta': beta
                }
            else:
                # No congestion observed
                newell_params[a] = {
                    'T0': 0, 'T3': 0, 'P': 0,
                    'mu': self.phase_params[a][0]['mu'], 'lambda': 0, 'beta': 0.1
                }
        
        return newell_params


class ModelZ:
    """Model Z: Vehicle Routing using NumPy arrays"""
    
    def __init__(self, nodes_df: pd.DataFrame, links_df: pd.DataFrame, network_graph: nx.DiGraph, params: NetworkParams):
        self.nodes_df = nodes_df
        self.links_df = links_df
        self.network_graph = network_graph
        self.params = params
        
        # Identify zones
        if 'zone_id' in nodes_df.columns:
            self.zones = nodes_df[nodes_df['zone_id'] > 0]['node_id'].tolist()
        else:
            self.zones = []
        self.num_zones = len(self.zones)
        
        # Generate OD pairs and find paths
        self._generate_od_pairs()
        self._find_shortest_paths()
        
        print(f"Model Z initialized with {self.num_zones} zones, {len(self.od_pairs)} OD pairs")
    
    def _generate_od_pairs(self):
        """Generate OD pairs from zones"""
        self.od_pairs = []
        for origin in self.zones:
            for destination in self.zones:
                if origin != destination:
                    self.od_pairs.append((origin, destination))
    
    def _find_shortest_paths(self):
        """Find shortest paths for all OD pairs"""
        self.paths = {}
        self.path_links = {}
        
        for origin, destination in self.od_pairs:
            try:
                # Find shortest path by travel time
                path = nx.shortest_path(
                    self.network_graph, 
                    origin, 
                    destination, 
                    weight='fftt'
                )
                
                # Convert path to link sequence
                links = []
                for i in range(len(path) - 1):
                    edge_data = self.network_graph[path[i]][path[i+1]]
                    links.append(edge_data['link_id'])
                
                self.paths[(origin, destination)] = path
                self.path_links[(origin, destination)] = links
                
            except nx.NetworkXNoPath:
                print(f"No path found from {origin} to {destination}")
    
    def generate_od_demand(self, peak_hour_factor: float = 1.0) -> Dict:
        """Generate synthetic OD demand"""
        od_demand = {}
        
        for origin, destination in self.od_pairs:
            # Simple gravity model
            try:
                path_length = sum(
                    self.network_graph[self.paths[(origin, destination)][i]]
                    [self.paths[(origin, destination)][i+1]]['length']
                    for i in range(len(self.paths[(origin, destination)]) - 1)
                )
                
                # Inverse relationship with distance
                base_demand = max(10, 1000 / (1 + path_length / 1000))
                demand = base_demand * peak_hour_factor * np.random.uniform(0.5, 1.5)
                od_demand[(origin, destination)] = demand
                
            except KeyError:
                od_demand[(origin, destination)] = 0
        
        return od_demand
    
    def assign_flows_to_network(self, od_demand: Dict, travel_times: np.ndarray) -> np.ndarray:
        """Assign OD flows to network links"""
        link_flows = np.zeros((len(self.links_df), self.params.time_horizon))
        
        # Create link ID to index mapping
        link_id_to_idx = {row['link_id']: idx for idx, row in self.links_df.iterrows()}
        
        # Temporal demand distribution
        peak_time = self.params.time_horizon // 3
        time_factors = np.exp(-0.5 * ((np.arange(self.params.time_horizon) - peak_time) / 10)**2)
        time_factors = time_factors / np.sum(time_factors)
        
        for (origin, destination), demand in od_demand.items():
            if demand > 0 and (origin, destination) in self.path_links:
                links = self.path_links[(origin, destination)]
                
                # Distribute demand over time
                for t in range(self.params.time_horizon):
                    time_demand = demand * time_factors[t] * self.params.demand_multiplier
                    
                    # Add flow to each link in the path
                    for link_id in links:
                        if link_id in link_id_to_idx:
                            link_idx = link_id_to_idx[link_id]
                            link_flows[link_idx, t] += time_demand
        
        return link_flows


class PhaseAugmentedDTA:
    """Main solver using only NumPy arrays"""
    
    def __init__(self, data_loader: NetworkDataLoader, params: NetworkParams):
        self.data_loader = data_loader
        self.params = params
        
        # Initialize models
        self.model_x = ModelX(data_loader.links_df, params)
        self.model_y = ModelY(data_loader.links_df, params)
        self.model_z = ModelZ(
            data_loader.nodes_df, 
            data_loader.links_df, 
            data_loader.network_graph, 
            params
        )
        
        # Solution parameters
        self.max_iterations = 15
        self.convergence_threshold = 1e-3
        
        # Results storage
        self.results_history = {
            'iteration': [],
            'total_delay': [],
            'total_travel_time': [],
            'max_queue': [],
            'convergence_measure': []
        }
        
        print("NumPy-based Phase-Augmented DTA initialized successfully")
    
    def solve(self) -> Dict:
        """Main solution algorithm"""
        print("=" * 60)
        print("STARTING PHASE-AUGMENTED DTA SOLUTION (NumPy Version)")
        print("=" * 60)
        
        # Generate initial OD demand
        print("Step 0: Generating OD demand...")
        od_demand = self.model_z.generate_od_demand(peak_hour_factor=1.2)
        total_demand = sum(od_demand.values())
        print(f"Total OD demand: {total_demand:.0f} vehicles")
        
        # Initialize with free-flow conditions
        initial_travel_times = self.model_x.free_flow_time.copy()
        current_inflows = np.zeros((self.model_x.A, self.model_x.T))
        
        print(f"\nIterative solution (max {self.max_iterations} iterations):")
        print("-" * 60)
        
        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1:2d}]")
            
            # Step 1: Vehicle routing (Model Z)
            print("  → Vehicle routing and flow assignment...")
            new_inflows = self.model_z.assign_flows_to_network(od_demand, initial_travel_times)
            
            # Step 2: Phase selection (Model Y)
            print("  → Phase selection and control...")
            phase_selections = self.model_y.select_phases(
                self.model_x.x, 
                new_inflows
            )
            discharge_rates = self.model_y.get_discharge_rates()
            
            # Step 3: Queue dynamics update (Model X)
            print("  → Queue dynamics update...")
            queue_results = self.model_x.update_queue_dynamics(new_inflows, discharge_rates)
            
            # Update travel times
            new_travel_times = self.model_x.calculate_travel_times()
            
            # Calculate performance metrics
            total_delay = np.sum(queue_results['queue'])
            total_travel_time = np.sum(new_travel_times * np.sum(new_inflows, axis=1))
            max_queue = np.max(queue_results['queue'])
            
            # Check convergence
            if iteration > 0:
                convergence_measure = np.linalg.norm(new_travel_times - initial_travel_times) / np.linalg.norm(initial_travel_times)
            else:
                convergence_measure = float('inf')
            
            # Store results
            self.results_history['iteration'].append(iteration + 1)
            self.results_history['total_delay'].append(total_delay)
            self.results_history['total_travel_time'].append(total_travel_time)
            self.results_history['max_queue'].append(max_queue)
            self.results_history['convergence_measure'].append(convergence_measure)
            
            # Print iteration results
            print(f"     Total delay: {total_delay:8.1f} veh·min")
            print(f"     Total travel time: {total_travel_time:8.1f} veh·min")
            print(f"     Max queue length: {max_queue:8.1f} vehicles")
            print(f"     Convergence measure: {convergence_measure:8.6f}")
            
            # Check convergence
            if convergence_measure < self.convergence_threshold:
                print(f"  ✓ Converged at iteration {iteration + 1}")
                break
            
            # Update for next iteration
            alpha = 0.3  # Step size for averaging
            initial_travel_times = (1 - alpha) * initial_travel_times + alpha * new_travel_times
            current_inflows = new_inflows
        
        print("\n" + "=" * 60)
        print("SOLUTION COMPLETED")
        print("=" * 60)
        
        return self._prepare_final_results(queue_results, phase_selections, new_travel_times)
    
    def _prepare_final_results(self, queue_results: Dict, phase_selections: np.ndarray, travel_times: np.ndarray) -> Dict:
        """Prepare comprehensive results"""
        
        # Link-level results
        link_results = []
        for idx, (_, link) in enumerate(self.data_loader.links_df.iterrows()):
            avg_queue = np.mean(queue_results['queue'][idx, :])
            avg_inflow = np.mean(queue_results['inflow'][idx, :])
            avg_outflow = np.mean(queue_results['outflow'][idx, :])
            travel_time = travel_times[idx]
            
            # Dominant phase
            dominant_phase = np.argmax(phase_selections[idx, :])
            
            link_results.append({
                'link_id': link['link_id'],
                'from_node': link['from_node_id'],
                'to_node': link['to_node_id'],
                'avg_queue': avg_queue,
                'avg_inflow_per_min': avg_inflow,
                'avg_outflow_per_min': avg_outflow,
                'travel_time_min': travel_time,
                'dominant_phase': dominant_phase,
                'free_flow_time': self.model_x.free_flow_time[idx],
                'delay_ratio': travel_time / max(self.model_x.free_flow_time[idx], 0.1)
            })
        
        # Phase selection summary
        phase_summary = {}
        for p in range(self.params.num_phases):
            phase_usage = np.mean(phase_selections[:, p])
            phase_summary[f'phase_{p}_usage'] = phase_usage
        
        # Newell parameters
        newell_params = self.model_y.estimate_newell_parameters(queue_results)
        
        return {
            'convergence_history': self.results_history,
            'final_performance': {
                'total_delay': self.results_history['total_delay'][-1],
                'total_travel_time': self.results_history['total_travel_time'][-1],
                'max_queue': self.results_history['max_queue'][-1],
                'avg_travel_time': np.mean(travel_times),
                'network_efficiency': np.sum(self.model_x.free_flow_time) / np.sum(travel_times)
            },
            'link_results': link_results,
            'phase_summary': phase_summary,
            'newell_parameters': newell_params,
            'raw_data': {
                'queue_evolution': queue_results['queue'],
                'phase_selections': phase_selections,
                'travel_times': travel_times
            }
        }
    
    def export_results(self, results: Dict, output_prefix: str = "pa_dta_results"):
        """Export results to CSV files"""
        
        # Export link performance
        link_df = pd.DataFrame(results['link_results'])
        link_df.to_csv(f"{output_prefix}_link_performance.csv", index=False)
        
        # Export convergence history
        conv_df = pd.DataFrame(results['convergence_history'])
        conv_df.to_csv(f"{output_prefix}_convergence.csv", index=False)
        
        # Export phase usage summary
        phase_df = pd.DataFrame([results['phase_summary']])
        phase_df.to_csv(f"{output_prefix}_phase_summary.csv", index=False)
        
        # Export performance summary
        perf_df = pd.DataFrame([results['final_performance']])
        perf_df.to_csv(f"{output_prefix}_performance_summary.csv", index=False)
        
        print(f"Results exported to {output_prefix}_*.csv files")
    
    def visualize_results(self, results: Dict):
        """Create visualization using matplotlib"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase-Augmented DTA Results Analysis (NumPy Version)', fontsize=16, fontweight='bold')
        
        # Plot 1: Convergence history
        conv_data = results['convergence_history']
        axes[0, 0].plot(conv_data['iteration'], conv_data['total_delay'], 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_title('Convergence: Total System Delay')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Total Delay (veh·min)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Travel time comparison
        link_data = pd.DataFrame(results['link_results'])
        x_pos = range(len(link_data))
        axes[0, 1].bar(x_pos, link_data['free_flow_time'], alpha=0.6, label='Free-flow', color='green')
        axes[0, 1].bar(x_pos, link_data['travel_time_min'], alpha=0.8, label='Actual', color='red')
        axes[0, 1].set_title('Travel Times by Link')
        axes[0, 1].set_xlabel('Link Index')
        axes[0, 1].set_ylabel('Travel Time (min)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Queue evolution for top congested links
        queue_data = results['raw_data']['queue_evolution']
        top_congested = link_data.nlargest(3, 'avg_queue')
        colors = ['red', 'blue', 'green']
        
        for i, (_, link_row) in enumerate(top_congested.iterrows()):
            if i < 3:  # Only plot top 3
                link_idx = link_row.name  # DataFrame index corresponds to link index
                time_steps = range(queue_data.shape[1])
                axes[0, 2].plot(time_steps, queue_data[link_idx, :], 
                              color=colors[i], linewidth=2, label=f'Link {link_row["link_id"]}')
        
        axes[0, 2].set_title('Queue Evolution - Top 3 Congested Links')
        axes[0, 2].set_xlabel('Time (min)')
        axes[0, 2].set_ylabel('Queue Length (veh)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Phase usage distribution
        phase_summary = results['phase_summary']
        phase_names = [f'Phase {i}' for i in range(len(phase_summary))]
        phase_values = list(phase_summary.values())
        colors_phase = plt.cm.Set3(np.linspace(0, 1, len(phase_values)))
        
        wedges, texts, autotexts = axes[1, 0].pie(phase_values, labels=phase_names, autopct='%1.1f%%', 
                                                 colors=colors_phase, startangle=90)
        axes[1, 0].set_title('Phase Usage Distribution')
        
        # Plot 5: Link performance scatter
        scatter = axes[1, 1].scatter(link_data['avg_inflow_per_min'], link_data['avg_queue'], 
                                   c=link_data['delay_ratio'], cmap='YlOrRd', s=60, alpha=0.7)
        axes[1, 1].set_xlabel('Average Inflow (veh/min)')
        axes[1, 1].set_ylabel('Average Queue (veh)')
        axes[1, 1].set_title('Link Performance: Inflow vs Queue')
        cbar = plt.colorbar(scatter, ax=axes[1, 1])
        cbar.set_label('Delay Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Performance metrics
        metrics = results['final_performance']
        metric_names = ['Total Delay\n(veh·min)', 'Max Queue\n(veh)', 'Avg Travel Time\n(min)', 
                       'Network\nEfficiency']
        metric_values = [metrics['total_delay'], results['convergence_history']['max_queue'][-1], 
                        metrics['avg_travel_time'], metrics['network_efficiency']]
        
        # Normalize values for display
        max_val = max(metric_values)
        normalized_values = [v/max_val for v in metric_values]
        bars = axes[1, 2].bar(metric_names, normalized_values, color=['red', 'orange', 'blue', 'green'], alpha=0.7)
        axes[1, 2].set_title('Final Performance Metrics (Normalized)')
        axes[1, 2].set_ylabel('Normalized Value')
        
        # Add actual values as text on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_detailed_summary(self, results: Dict):
        """Print detailed summary of results"""
        
        print("\n" + "="*80)
        print("PHASE-AUGMENTED DTA - DETAILED RESULTS SUMMARY (NumPy Version)")
        print("="*80)
        
        # Network statistics
        net_stats = self.data_loader.get_network_stats()
        print(f"\nNETWORK CHARACTERISTICS:")
        print(f"  • Nodes: {net_stats['num_nodes']}")
        print(f"  • Links: {net_stats['num_links']}")
        print(f"  • Zones: {net_stats['num_zones']}")
        print(f"  • Total capacity: {net_stats['total_capacity']:.0f} veh/h")
        print(f"  • Average link length: {net_stats['avg_link_length']:.1f} units")
        print(f"  • Total network length: {net_stats['total_network_length']:.1f} units")
        
        # Solution performance
        perf = results['final_performance']
        conv = results['convergence_history']
        print(f"\nSOLUTION PERFORMANCE:")
        print(f"  • Iterations to convergence: {len(conv['iteration'])}")
        print(f"  • Final convergence measure: {conv['convergence_measure'][-1]:.2e}")
        print(f"  • Total system delay: {perf['total_delay']:.1f} veh·min")
        print(f"  • Total travel time: {perf['total_travel_time']:.1f} veh·min")
        print(f"  • Average travel time: {perf['avg_travel_time']:.2f} min")
        print(f"  • Maximum queue length: {perf['max_queue']:.1f} vehicles")
        print(f"  • Network efficiency: {perf['network_efficiency']:.3f}")
        
        # Phase utilization
        phase_summary = results['phase_summary']
        print(f"\nPHASE UTILIZATION:")
        phase_names = ['Free Flow', 'Light Congestion', 'Heavy Congestion', 'Breakdown']
        for i, (phase, usage) in enumerate(phase_summary.items()):
            phase_num = int(phase.split('_')[1])
            phase_name = phase_names[phase_num] if phase_num < len(phase_names) else f'Phase {phase_num}'
            print(f"  • Phase {phase_num} ({phase_name}): {usage:.1%}")
        
        # Top congested links
        link_data = pd.DataFrame(results['link_results'])
        top_congested = link_data.nlargest(5, 'avg_queue')
        print(f"\nTOP 5 CONGESTED LINKS:")
        for _, link in top_congested.iterrows():
            print(f"  • Link {link['link_id']} ({link['from_node']}→{link['to_node']}): "
                  f"{link['avg_queue']:.1f} veh queue, {link['delay_ratio']:.2f}x delay")
        
        # Model-specific insights
        print(f"\nMODEL-SPECIFIC INSIGHTS:")
        
        # Model X insights
        total_vehicle_hours = np.sum(results['raw_data']['queue_evolution'])
        print(f"  Model X (Queue Dynamics):")
        print(f"    - Total vehicle-hours in queues: {total_vehicle_hours:.1f}")
        print(f"    - Average system occupancy: {total_vehicle_hours / (len(link_data) * self.params.time_horizon):.2f} veh/link")
        
        # Model Y insights
        phase_selections = results['raw_data']['phase_selections']
        mixed_phase_links = 0
        total_phase_diversity = 0
        
        for a in range(phase_selections.shape[0]):
            max_usage = np.max(phase_selections[a, :])
            if max_usage < 0.8:  # No strongly dominant phase
                mixed_phase_links += 1
            # Calculate diversity (standard deviation of phase probabilities)
            total_phase_diversity += np.std(phase_selections[a, :])
        
        avg_phase_diversity = total_phase_diversity / phase_selections.shape[0]
        
        print(f"  Model Y (Phase Control):")
        print(f"    - Links with mixed phase usage: {mixed_phase_links}/{len(link_data)} ({mixed_phase_links/len(link_data)*100:.1f}%)")
        print(f"    - Average phase diversity: {avg_phase_diversity:.3f}")
        
        # Model Z insights
        total_flow = np.sum(results['raw_data']['queue_evolution'])
        print(f"  Model Z (Vehicle Routing):")
        print(f"    - Total vehicle-trips processed: {total_flow:.0f}")
        if len(self.model_z.od_pairs) > 0:
            print(f"    - Average flow per OD pair: {total_flow / len(self.model_z.od_pairs):.1f}")
        
        print("\n" + "="*80)


def generate_sample_data():
    """Generate sample network data for testing if files are not available"""
    
    print("Generating sample network data for testing...")
    
    # Sample nodes (simplified Sioux Falls-like network)
    nodes_data = {
        'name': [1, 2, 3, 4, 5, 6, 7, 8],
        'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'zone_id': [1, 2, 3, 4, 0, 0, 0, 0],  # First 4 are zones
        'x_coord': [0, 100, 200, 100, 50, 150, 75, 125],
        'y_coord': [0, 0, 0, 100, 50, 50, 75, 75],
        'district_id': [1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    # Sample links
    links_data = {
        'name': list(range(1, 17)),
        'link_id': list(range(1, 17)),
        'from_node_id': [1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8],
        'to_node_id': [5, 6, 7, 8, 2, 3, 4, 1, 6, 7, 8, 5, 1, 2, 3, 4],
        'dir_flag': [1] * 16,
        'length': [50, 50, 50, 50, 100, 100, 100, 100, 70, 70, 70, 70, 60, 60, 60, 60],
        'lanes': [2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2],
        'capacity': [1800, 1800, 1800, 1800, 2700, 2700, 2700, 2700, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800],
        'free_speed': [45, 45, 45, 45, 55, 55, 55, 55, 40, 40, 40, 40, 35, 35, 35, 35],
        'link_type': [1] * 16,
        'VDF_alpha': [0.15] * 16,
        'VDF_beta': [4] * 16,
        'ref_volume': [800, 900, 700, 850, 1200, 1100, 950, 1000, 600, 650, 580, 620, 500, 550, 480, 520],
        'base_volume': [0] * 16,
        'base_vol_auto': [0] * 16,
        'restricted_turn_nodes': [0] * 16,
        'VDF_fftt': [1.33, 1.33, 1.33, 1.33, 1.82, 1.82, 1.82, 1.82, 1.75, 1.75, 1.75, 1.75, 1.71, 1.71, 1.71, 1.71],
        'VDF_toll_auto': [0] * 16,
        'geometry': [''] * 16
    }
    
    # Create DataFrames
    nodes_df = pd.DataFrame(nodes_data)
    links_df = pd.DataFrame(links_data)
    
    # Save to CSV files
    nodes_df.to_csv('sample_node.csv', index=False)
    links_df.to_csv('sample_link.csv', index=False)
    
    # Create empty route assignment file
    routes_df = pd.DataFrame()
    routes_df.to_csv('sample_route_assignment.csv', index=False)
    
    print("Sample data files created: sample_node.csv, sample_link.csv, sample_route_assignment.csv")
    
    return 'sample_node.csv', 'sample_link.csv', 'sample_route_assignment.csv'


def main():
    """Main execution function"""
    
    print("Phase-Augmented Dynamic Traffic Assignment")
    print("NumPy-Only Implementation")
    print("="*50)
    
    # Initialize data loader
    data_loader = NetworkDataLoader()
    
    # Try to load network data, create sample if not available
    node_file = 'node.csv'
    link_file = 'link.csv'
    route_file = 'route_assignment.csv'
    
    success = data_loader.load_data(node_file, link_file, route_file)
    
    if not success:
        print("Original data files not found. Generating sample data...")
        node_file, link_file, route_file = generate_sample_data()
        success = data_loader.load_data(node_file, link_file, route_file)
        
        if not success:
            print("Failed to load data. Please check file availability.")
            return
    
    # Print network statistics
    stats = data_loader.get_network_stats()
    print(f"\nNetwork loaded successfully:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Set up parameters
    params = NetworkParams(
        time_horizon=90,  # 1.5 hours
        delta_t=1.0,      # 1-minute time steps
        num_phases=4,     # 4 operational phases
        demand_multiplier=1.5  # Scale up demand for testing
    )
    
    # Create and run solver
    print(f"\nInitializing PA-DTA solver (NumPy version)...")
    solver = PhaseAugmentedDTA(data_loader, params)
    
    # Solve the problem
    start_time = time.time()
    results = solver.solve()
    solution_time = time.time() - start_time
    
    print(f"\nSolution completed in {solution_time:.2f} seconds")
    
    # Print detailed summary
    solver.print_detailed_summary(results)
    
    # Export results
    solver.export_results(results, "numpy_pa_dta")
    
    # Visualize results
    try:
        solver.visualize_results(results)
    except Exception as e:
        print(f"Visualization error (non-critical): {e}")
    
    # Generate additional analysis
    print("\nGenerating additional analysis...")
    
    # Time-series analysis
    print("\nTime-series analysis of key metrics:")
    queue_data = results['raw_data']['queue_evolution']
    
    # Find peak congestion time
    total_queue_by_time = np.sum(queue_data, axis=0)
    peak_time = np.argmax(total_queue_by_time)
    peak_queue = total_queue_by_time[peak_time]
    
    print(f"  • Peak congestion time: {peak_time} minutes")
    print(f"  • Peak total queue: {peak_queue:.1f} vehicles")
    print(f"  • Congestion duration: {np.sum(total_queue_by_time > peak_queue * 0.5)} minutes")
    
    # Efficiency metrics
    link_data = pd.DataFrame(results['link_results'])
    congested_links = len(link_data[link_data['delay_ratio'] > 1.2])
    print(f"  • Significantly congested links: {congested_links}/{len(link_data)} ({congested_links/len(link_data)*100:.1f}%)")
    
    # Memory usage information
    queue_memory = queue_data.nbytes / (1024 * 1024)  # MB
    print(f"  • Memory usage for queue data: {queue_memory:.2f} MB")
    
    return solver, results


# Additional utility functions
def save_results_to_json(results: Dict, filename: str = "pa_dta_results.json"):
    """Save results to JSON file for further analysis"""
    
    # Convert NumPy arrays to lists for JSON serialization
    json_results = {}
    
    for key, value in results.items():
        if key == 'raw_data':
            json_results[key] = {
                'queue_evolution': value['queue_evolution'].tolist(),
                'phase_selections': value['phase_selections'].tolist(),
                'travel_times': value['travel_times'].tolist()
            }
        elif isinstance(value, dict):
            json_results[key] = value
        else:
            json_results[key] = value
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {filename}")


def analyze_phase_transitions(phase_selections: np.ndarray) -> Dict:
    """Analyze phase transition patterns"""
    
    A, P = phase_selections.shape
    transitions = np.zeros((P, P))  # Transition matrix
    
    # Count dominant phase for each link
    dominant_phases = np.argmax(phase_selections, axis=1)
    
    # Analyze phase distribution
    phase_distribution = {}
    for p in range(P):
        count = np.sum(dominant_phases == p)
        phase_distribution[f'phase_{p}'] = count / A
    
    return {
        'phase_distribution': phase_distribution,
        'dominant_phases': dominant_phases.tolist(),
        'phase_entropy': -np.sum(phase_selections * np.log(phase_selections + 1e-10), axis=1).mean()
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the main analysis
    solver, results = main()
    
    print("\n" + "="*80)
    print("NUMPY-BASED PHASE-AUGMENTED DTA ANALYSIS COMPLETED")
    print("="*80)
    print("\nKey advantages of this NumPy implementation:")
    print("  ✓ No external dependencies beyond standard scientific Python")
    print("  ✓ Easier to understand and modify")
    print("  ✓ Compatible with any Python environment")
    print("  ✓ Memory efficient for medium-sized networks")
    print("  ✓ Straightforward debugging and profiling")
    
    print("\nOutput files generated:")
    print("  • numpy_pa_dta_link_performance.csv - Link-level results")
    print("  • numpy_pa_dta_convergence.csv - Convergence history")  
    print("  • numpy_pa_dta_phase_summary.csv - Phase usage statistics")
    print("  • numpy_pa_dta_performance_summary.csv - Overall performance")
    
    # Save additional JSON results
    save_results_to_json(results, "numpy_pa_dta_detailed.json")
    
    # Analyze phase transitions
    phase_analysis = analyze_phase_transitions(results['raw_data']['phase_selections'])
    print(f"\nPhase Analysis:")
    print(f"  • Phase entropy (diversity): {phase_analysis['phase_entropy']:.3f}")
    for phase, prob in phase_analysis['phase_distribution'].items():
        print(f"  • {phase} dominance: {prob:.1%}")
    
    print("\nImplementation successfully demonstrates:")
    print("  1. Pure NumPy implementation of phase-augmented DTA")
    print("  2. Integration of Models X, Y, and Z without tensors")
    print("  3. Real network data processing and analysis")
    print("  4. Comprehensive results export and visualization")
    print("  5. Memory-efficient handling of temporal dynamics")