# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 19:24:45 2025

@author: xzhou
"""

#!/usr/bin/env python3
"""
Model X (Queue Dynamics) Test Suite
Testing individual components with known inputs and expected outputs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time


class ModelX:
    """Model X: Queue Dynamics - Standalone Testing Version"""
    
    def __init__(self, links_df: pd.DataFrame, time_horizon: int = 60):
        self.links_df = links_df
        self.A = len(links_df)  # Number of links
        self.T = time_horizon   # Time horizon
        
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
        print(f"Capacity range: {self.capacity.min():.0f} - {self.capacity.max():.0f} veh/h")
        print(f"Free-flow time range: {self.free_flow_time.min():.1f} - {self.free_flow_time.max():.1f} min")
    
    def update_queue_dynamics(self, inflow: np.ndarray, discharge_rates: np.ndarray, verbose: bool = False) -> Dict:
        """
        Update queue dynamics with detailed tracking
        
        Args:
            inflow: (A, T) array of inflow rates [veh/min]
            discharge_rates: (A, T) array of discharge rates [veh/min]
            verbose: Print detailed step-by-step information
            
        Returns:
            Dictionary with queue evolution results
        """
        if verbose:
            print("\n" + "="*60)
            print("QUEUE DYNAMICS UPDATE - DETAILED TRACE")
            print("="*60)
        
        self.u = inflow.copy()
        
        # Reset state arrays
        self.x.fill(0.0)
        self.g.fill(0.0)
        
        # Store intermediate results for analysis
        diagnostics = {
            'effective_capacity': np.zeros((self.A, self.T)),
            'available_vehicles': np.zeros((self.A, self.T)),
            'capacity_utilization': np.zeros((self.A, self.T)),
            'queue_change': np.zeros((self.A, self.T))
        }
        
        for t in range(self.T - 1):
            if verbose and t < 5:  # Print first 5 time steps
                print(f"\nTime step {t}:")
            
            # Calculate effective discharge capacity (minimum of phase rates and link capacity)
            effective_capacity = np.minimum(discharge_rates[:, t], self.capacity_per_min)
            diagnostics['effective_capacity'][:, t] = effective_capacity
            
            # Calculate available vehicles (current queue + new arrivals)
            available_vehicles = self.x[:, t] + self.u[:, t]
            diagnostics['available_vehicles'][:, t] = available_vehicles
            
            # Calculate actual outflow (cannot exceed available vehicles or capacity)
            self.g[:, t] = np.minimum(available_vehicles, effective_capacity)
            
            # Calculate capacity utilization
            capacity_util = np.divide(self.g[:, t], effective_capacity, 
                                    out=np.zeros_like(self.g[:, t]), 
                                    where=effective_capacity>0)
            diagnostics['capacity_utilization'][:, t] = capacity_util
            
            # Update queue length using flow conservation
            self.x[:, t + 1] = self.x[:, t] + self.u[:, t] - self.g[:, t]
            
            # Ensure non-negative queues
            self.x[:, t + 1] = np.maximum(self.x[:, t + 1], 0.0)
            
            # Calculate queue change
            diagnostics['queue_change'][:, t] = self.x[:, t + 1] - self.x[:, t]
            
            if verbose and t < 5:
                for a in range(min(3, self.A)):  # Show first 3 links
                    print(f"  Link {a}: Queue={self.x[a,t]:6.2f} + Inflow={self.u[a,t]:6.2f} "
                          f"- Outflow={self.g[a,t]:6.2f} = New_Queue={self.x[a,t+1]:6.2f}")
        
        if verbose:
            print(f"\nFinal Summary:")
            print(f"  Total vehicles entered: {np.sum(self.u):.1f}")
            print(f"  Total vehicles exited: {np.sum(self.g):.1f}")
            print(f"  Vehicles remaining in queues: {np.sum(self.x[:, -1]):.1f}")
            print(f"  Max queue length: {np.max(self.x):.1f} vehicles")
            print(f"  Max queue link: {np.unravel_index(np.argmax(self.x), self.x.shape)}")
        
        return {
            'queue': self.x.copy(),
            'inflow': self.u.copy(),
            'outflow': self.g.copy(),
            'diagnostics': diagnostics
        }
    
    def calculate_travel_times(self, current_time: int = None, verbose: bool = False) -> np.ndarray:
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
        
        if verbose:
            print("\nTRAVEL TIME CALCULATION:")
            print("Link | Free-Flow | Volume/Cap | BPR Factor | Congestion | Queue Delay | Total")
            print("-" * 80)
            for a in range(min(5, self.A)):
                print(f"{a:4d} | {self.free_flow_time[a]:9.2f} | "
                      f"{volume_capacity_ratio[a]:10.3f} | {bpr_factor[a]:10.3f} | "
                      f"{congestion_time[a]:10.2f} | {queue_delay[a]:11.2f} | {total_travel_time[a]:5.2f}")
        
        return total_travel_time


def create_test_network() -> pd.DataFrame:
    """Create a simple test network for Model X testing"""
    
    # Create a simple 3-link test network
    test_links = {
        'link_id': [1, 2, 3],
        'from_node_id': [1, 2, 3],
        'to_node_id': [2, 3, 4],
        'length': [1000, 1500, 800],        # meters
        'capacity': [1800, 2400, 1200],      # veh/h
        'free_speed': [50, 60, 40],          # km/h
        'lanes': [2, 3, 2],
        'VDF_alpha': [0.15, 0.15, 0.15],
        'VDF_beta': [4.0, 4.0, 4.0],
        'VDF_fftt': [1.2, 1.5, 1.2]         # minutes
    }
    
    return pd.DataFrame(test_links)


def test_case_1_basic_flow_conservation():
    """Test Case 1: Basic Flow Conservation"""
    
    print("\n" + "="*80)
    print("TEST CASE 1: BASIC FLOW CONSERVATION")
    print("="*80)
    print("Objective: Verify that flow conservation equation works correctly")
    print("x[t+1] = x[t] + u[t] - g[t]")
    
    # Create test network
    links_df = create_test_network()
    model_x = ModelX(links_df, time_horizon=10)
    
    # Test Input: Simple constant inflow, unlimited capacity
    print("\nInput Conditions:")
    print("- 3 links with unlimited discharge capacity")
    print("- Constant inflow of 10 veh/min for first 5 time steps")
    print("- No inflow after t=5")
    
    # Create input arrays
    inflow = np.zeros((3, 10))
    inflow[:, 0:5] = 10.0  # 10 veh/min for first 5 time steps
    
    discharge_rates = np.ones((3, 10)) * 1000  # Very high capacity (unlimited)
    
    # Expected Output
    print("\nExpected Output:")
    print("- No queuing should occur (capacity >> inflow)")
    print("- Outflow should equal inflow at each time step")
    print("- All queues should remain at zero")
    
    # Run test
    results = model_x.update_queue_dynamics(inflow, discharge_rates, verbose=True)
    
    # Verify results
    print("\nVERIFICATION:")
    total_inflow = np.sum(results['inflow'])
    total_outflow = np.sum(results['outflow'])
    final_queues = np.sum(results['queue'][:, -1])
    
    print(f"Total inflow: {total_inflow:.1f} vehicles")
    print(f"Total outflow: {total_outflow:.1f} vehicles")
    print(f"Final queues: {final_queues:.1f} vehicles")
    print(f"Conservation check: {abs(total_inflow - total_outflow - final_queues) < 0.001}")
    
    # Test passes if conservation holds
    assert abs(total_inflow - total_outflow - final_queues) < 0.001, "Flow conservation violated!"
    print("✓ TEST CASE 1 PASSED")
    
    return results


def test_case_2_queue_formation():
    """Test Case 2: Queue Formation Under Capacity Constraints"""
    
    print("\n" + "="*80)
    print("TEST CASE 2: QUEUE FORMATION UNDER CAPACITY CONSTRAINTS")
    print("="*80)
    print("Objective: Verify queue formation when inflow exceeds capacity")
    
    # Create test network
    links_df = create_test_network()
    model_x = ModelX(links_df, time_horizon=20)
    
    print("\nInput Conditions:")
    print("- Link 0: High inflow (50 veh/min), Low capacity (20 veh/min)")
    print("- Link 1: Medium inflow (30 veh/min), Medium capacity (30 veh/min)")
    print("- Link 2: Low inflow (10 veh/min), High capacity (40 veh/min)")
    
    # Create input arrays
    inflow = np.zeros((3, 20))
    inflow[0, 5:15] = 50.0   # High inflow on link 0
    inflow[1, 5:15] = 30.0   # Medium inflow on link 1
    inflow[2, 5:15] = 10.0   # Low inflow on link 2
    
    discharge_rates = np.zeros((3, 20))
    discharge_rates[0, :] = 20.0   # Low capacity on link 0
    discharge_rates[1, :] = 30.0   # Medium capacity on link 1
    discharge_rates[2, :] = 40.0   # High capacity on link 2
    
    print("\nExpected Output:")
    print("- Link 0: Queue buildup (inflow > capacity)")
    print("- Link 1: No queuing (inflow = capacity)")
    print("- Link 2: No queuing (inflow < capacity)")
    
    # Run test
    results = model_x.update_queue_dynamics(inflow, discharge_rates, verbose=True)
    
    # Analyze results
    print("\nRESULTS ANALYSIS:")
    for link in range(3):
        max_queue = np.max(results['queue'][link, :])
        total_inflow = np.sum(results['inflow'][link, :])
        total_outflow = np.sum(results['outflow'][link, :])
        
        print(f"Link {link}:")
        print(f"  Max queue: {max_queue:.1f} vehicles")
        print(f"  Total inflow: {total_inflow:.1f} vehicles")
        print(f"  Total outflow: {total_outflow:.1f} vehicles")
        print(f"  Vehicles remaining: {total_inflow - total_outflow:.1f}")
    
    # Verification
    link_0_has_queue = np.max(results['queue'][0, :]) > 10
    link_1_minimal_queue = np.max(results['queue'][1, :]) < 5
    link_2_no_queue = np.max(results['queue'][2, :]) < 1
    
    print(f"\nVERIFICATION:")
    print(f"Link 0 has significant queue: {link_0_has_queue}")
    print(f"Link 1 has minimal queue: {link_1_minimal_queue}")
    print(f"Link 2 has no queue: {link_2_no_queue}")
    
    assert link_0_has_queue, "Link 0 should have queue buildup!"
    assert link_2_no_queue, "Link 2 should have no queue!"
    print("✓ TEST CASE 2 PASSED")
    
    return results


def test_case_3_realistic_scenario():
    """Test Case 3: Realistic Traffic Scenario"""
    
    print("\n" + "="*80)
    print("TEST CASE 3: REALISTIC TRAFFIC SCENARIO")
    print("="*80)
    print("Objective: Test with realistic traffic patterns and capacities")
    
    # Create test network with realistic parameters
    realistic_links = {
        'link_id': [101, 102, 103, 104],
        'from_node_id': [1, 2, 3, 4],
        'to_node_id': [2, 3, 4, 5],
        'length': [800, 1200, 600, 1000],
        'capacity': [1800, 2400, 1200, 1800],  # veh/h
        'free_speed': [50, 60, 40, 50],
        'lanes': [2, 3, 2, 2],
        'VDF_alpha': [0.15, 0.15, 0.15, 0.15],
        'VDF_beta': [4.0, 4.0, 4.0, 4.0],
        'VDF_fftt': [0.96, 1.2, 0.9, 1.2]
    }
    
    links_df = pd.DataFrame(realistic_links)
    model_x = ModelX(links_df, time_horizon=60)
    
    print("\nInput Conditions:")
    print("- 4 links with realistic highway capacities")
    print("- Morning peak demand pattern (bell curve)")
    print("- Phase-dependent discharge rates")
    
    # Create realistic demand pattern (morning peak)
    time_steps = np.arange(60)
    peak_time = 30
    demand_pattern = 0.8 * np.exp(-0.5 * ((time_steps - peak_time) / 10)**2)
    
    # Create inflow matrix with different demand levels per link
    inflow = np.zeros((4, 60))
    base_demands = [25, 35, 15, 30]  # veh/min at peak
    
    for link in range(4):
        inflow[link, :] = base_demands[link] * demand_pattern
    
    # Create phase-dependent discharge rates
    discharge_rates = np.zeros((4, 60))
    base_capacities = np.array([30, 40, 20, 30])  # veh/min
    
    # Simulate different phases: free-flow -> congestion -> recovery
    for t in range(60):
        if t < 20:  # Free flow phase
            phase_factor = 1.0
        elif t < 40:  # Congestion phase
            phase_factor = 0.8 - 0.2 * (t - 20) / 20  # Decreasing capacity
        else:  # Recovery phase
            phase_factor = 0.6 + 0.4 * (t - 40) / 20  # Recovering capacity
        
        discharge_rates[:, t] = base_capacities * phase_factor
    
    print(f"Peak demand time: {peak_time} minutes")
    print(f"Total demand: {np.sum(inflow):.0f} vehicles")
    print(f"Capacity varies from 60% to 100% of base capacity")
    
    # Run simulation
    results = model_x.update_queue_dynamics(inflow, discharge_rates, verbose=False)
    
    # Calculate travel times
    travel_times = model_x.calculate_travel_times(verbose=True)
    
    # Analyze results
    print(f"\nRESULTS ANALYSIS:")
    total_delay = np.sum(results['queue'])
    max_queue_time = np.unravel_index(np.argmax(results['queue']), results['queue'].shape)
    avg_travel_time = np.mean(travel_times)
    
    print(f"Total system delay: {total_delay:.1f} veh·min")
    print(f"Maximum queue: {np.max(results['queue']):.1f} vehicles at link {max_queue_time[0]}, time {max_queue_time[1]}")
    print(f"Average travel time: {avg_travel_time:.2f} minutes")
    print(f"Average free-flow time: {np.mean(model_x.free_flow_time):.2f} minutes")
    print(f"Average delay factor: {avg_travel_time / np.mean(model_x.free_flow_time):.2f}")
    
    # Test passes if results are reasonable
    assert total_delay > 0, "Should have some delay under congestion"
    assert avg_travel_time > np.mean(model_x.free_flow_time), "Travel time should exceed free-flow time"
    print("✓ TEST CASE 3 PASSED")
    
    return results, travel_times


def visualize_test_results(results_list: List[Dict], titles: List[str]):
    """Visualize test results"""
    
    fig, axes = plt.subplots(len(results_list), 3, figsize=(15, 5*len(results_list)))
    if len(results_list) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (results, title) in enumerate(zip(results_list, titles)):
        # Plot 1: Queue evolution
        for link in range(results['queue'].shape[0]):
            axes[i, 0].plot(results['queue'][link, :], label=f'Link {link}', linewidth=2)
        axes[i, 0].set_title(f'{title}: Queue Evolution')
        axes[i, 0].set_xlabel('Time (min)')
        axes[i, 0].set_ylabel('Queue Length (veh)')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot 2: Inflow vs Outflow
        total_inflow = np.sum(results['inflow'], axis=0)
        total_outflow = np.sum(results['outflow'], axis=0)
        
        axes[i, 1].plot(total_inflow, label='Total Inflow', linewidth=2, color='blue')
        axes[i, 1].plot(total_outflow, label='Total Outflow', linewidth=2, color='red')
        axes[i, 1].set_title(f'{title}: System Flow')
        axes[i, 1].set_xlabel('Time (min)')
        axes[i, 1].set_ylabel('Flow (veh/min)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative flow
        cum_inflow = np.cumsum(total_inflow)
        cum_outflow = np.cumsum(total_outflow)
        
        axes[i, 2].plot(cum_inflow, label='Cumulative Inflow', linewidth=2, color='blue')
        axes[i, 2].plot(cum_outflow, label='Cumulative Outflow', linewidth=2, color='red')
        axes[i, 2].fill_between(range(len(cum_inflow)), cum_outflow, cum_inflow, 
                               alpha=0.3, color='orange', label='Vehicles in System')
        axes[i, 2].set_title(f'{title}: Cumulative Flows')
        axes[i, 2].set_xlabel('Time (min)')
        axes[i, 2].set_ylabel('Cumulative Vehicles')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_all_tests():
    """Run all Model X tests"""
    
    print("MODEL X QUEUE DYNAMICS - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing individual Model X components before integration")
    
    # Run all test cases
    test_results = []
    test_titles = []
    
    try:
        # Test Case 1: Flow Conservation
        results1 = test_case_1_basic_flow_conservation()
        test_results.append(results1)
        test_titles.append("Flow Conservation")
        
        # Test Case 2: Queue Formation
        results2 = test_case_2_queue_formation()
        test_results.append(results2)
        test_titles.append("Queue Formation")
        
        # Test Case 3: Realistic Scenario
        results3, travel_times3 = test_case_3_realistic_scenario()
        test_results.append(results3)
        test_titles.append("Realistic Scenario")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        print("Model X is working correctly and ready for integration")
        
        # Visualize results
        print("\nGenerating visualization...")
        visualize_test_results(test_results, test_titles)
        
        return test_results, travel_times3
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return None, None
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        return None, None


def test_with_real_data(node_file='node.csv', link_file='link.csv'):
    """Test Model X with real network data"""
    
    print("\n" + "="*80)
    print("TEST CASE 4: REAL NETWORK DATA")
    print("="*80)
    print("Testing Model X with actual network files")
    
    try:
        # Load real network data
        links_df = pd.read_csv(link_file)
        print(f"Loaded {len(links_df)} links from {link_file}")
        
        # Initialize Model X with real data
        model_x = ModelX(links_df, time_horizon=60)
        
        # Create synthetic demand based on real network
        A = len(links_df)
        T = 60
        
        # Generate demand proportional to link capacity
        inflow = np.zeros((A, T))
        peak_time = 30
        
        for a in range(A):
            # Demand as fraction of capacity
            base_demand = links_df.iloc[a]['capacity'] / 60 * 0.3  # 30% of capacity
            demand_pattern = np.exp(-0.5 * ((np.arange(T) - peak_time) / 8)**2)
            inflow[a, :] = base_demand * demand_pattern
        
        # Phase-dependent discharge rates
        discharge_rates = np.zeros((A, T))
        for a in range(A):
            base_capacity = links_df.iloc[a]['capacity'] / 60
            # Simulate congestion: capacity drops during peak
            for t in range(T):
                if 20 <= t <= 40:  # Peak period
                    capacity_factor = 0.7 + 0.3 * np.random.random()  # 70-100% capacity
                else:
                    capacity_factor = 0.9 + 0.1 * np.random.random()  # 90-100% capacity
                
                discharge_rates[a, t] = base_capacity * capacity_factor
        
        print(f"Generated demand for {A} links over {T} time steps")
        print(f"Peak demand time: {peak_time} minutes")
        print(f"Total system demand: {np.sum(inflow):.0f} vehicles")
        
        # Run simulation
        results = model_x.update_queue_dynamics(inflow, discharge_rates, verbose=False)
        travel_times = model_x.calculate_travel_times(verbose=False)
        
        # Analyze results
        total_delay = np.sum(results['queue'])
        max_queue = np.max(results['queue'])
        avg_travel_time = np.mean(travel_times)
        congested_links = np.sum(np.max(results['queue'], axis=1) > 1)
        
        print(f"\nRESULTS FOR REAL NETWORK:")
        print(f"Total system delay: {total_delay:.1f} veh·min")
        print(f"Maximum queue length: {max_queue:.1f} vehicles")
        print(f"Average travel time: {avg_travel_time:.2f} minutes")
        print(f"Congested links: {congested_links}/{A} ({congested_links/A*100:.1f}%)")
        print(f"Network efficiency: {np.mean(model_x.free_flow_time)/avg_travel_time:.3f}")
        
        # Export results for further analysis
        link_results = []
        for a in range(A):
            link_results.append({
                'link_id': links_df.iloc[a]['link_id'],
                'max_queue': np.max(results['queue'][a, :]),
                'avg_inflow': np.mean(results['inflow'][a, :]),
                'avg_outflow': np.mean(results['outflow'][a, :]),
                'travel_time': travel_times[a],
                'free_flow_time': model_x.free_flow_time[a],
                'delay_ratio': travel_times[a] / model_x.free_flow_time[a]
            })
        
        results_df = pd.DataFrame(link_results)
        results_df.to_csv('model_x_test_results.csv', index=False)
        print(f"Detailed results exported to model_x_test_results.csv")
        
        print("✓ REAL DATA TEST COMPLETED SUCCESSFULLY")
        
        return results, travel_times, results_df
        
    except FileNotFoundError as e:
        print(f"Real data files not found: {e}")
        print("Using synthetic test network instead")
        return None, None, None
    except Exception as e:
        print(f"Error testing with real data: {e}")
        return None, None, None


def performance_benchmark():
    """Benchmark Model X performance with different network sizes"""
    
    print("\n" + "="*80)
    print("MODEL X PERFORMANCE BENCHMARK")
    print("="*80)
    
    network_sizes = [10, 50, 100, 500]
    time_horizons = [60, 120]
    
    results = []
    
    for num_links in network_sizes:
        for time_horizon in time_horizons:
            # Create synthetic network
            synthetic_links = {
                'link_id': list(range(1, num_links + 1)),
                'from_node_id': list(range(1, num_links + 1)),
                'to_node_id': list(range(2, num_links + 2)),
                'length': np.random.randint(500, 2000, num_links),
                'capacity': np.random.randint(1200, 2400, num_links),
                'free_speed': np.random.randint(40, 80, num_links),
                'lanes': np.random.randint(2, 4, num_links),
                'VDF_alpha': [0.15] * num_links,
                'VDF_beta': [4.0] * num_links,
                'VDF_fftt': np.random.uniform(0.5, 3.0, num_links)
            }
            
            links_df = pd.DataFrame(synthetic_links)
            
            # Initialize Model X
            model_x = ModelX(links_df, time_horizon=time_horizon)
            
            # Create random input data
            inflow = np.random.exponential(10, (num_links, time_horizon))
            discharge_rates = np.random.uniform(20, 50, (num_links, time_horizon))
            
            # Benchmark execution time
            start_time = time.time()
            results_sim = model_x.update_queue_dynamics(inflow, discharge_rates, verbose=False)
            travel_times = model_x.calculate_travel_times(verbose=False)
            end_time = time.time()
            
            execution_time = end_time - start_time
            memory_usage = (results_sim['queue'].nbytes + results_sim['inflow'].nbytes + 
                          results_sim['outflow'].nbytes) / (1024 * 1024)  # MB
            
            results.append({
                'num_links': num_links,
                'time_horizon': time_horizon,
                'execution_time': execution_time,
                'memory_usage_mb': memory_usage,
                'operations_per_second': (num_links * time_horizon) / execution_time
            })
            
            print(f"Network: {num_links:3d} links, {time_horizon:3d} time steps | "
                  f"Time: {execution_time:6.3f}s | Memory: {memory_usage:5.1f}MB | "
                  f"Ops/sec: {(num_links * time_horizon) / execution_time:8.0f}")
    
    # Create benchmark summary
    benchmark_df = pd.DataFrame(results)
    benchmark_df.to_csv('model_x_benchmark.csv', index=False)
    print(f"\nBenchmark results saved to model_x_benchmark.csv")
    
    return benchmark_df


def export_test_inputs_outputs():
    """Export specific test inputs and expected outputs for documentation"""
    
    print("\n" + "="*80)
    print("EXPORTING TEST INPUTS AND EXPECTED OUTPUTS")
    print("="*80)
    
    # Test Case 1 Data
    test1_data = {
        'description': 'Basic Flow Conservation Test',
        'network': {
            'num_links': 3,
            'capacities_veh_per_hour': [1800, 2400, 1200],
            'free_flow_times_min': [1.2, 1.5, 1.2]
        },
        'inputs': {
            'time_horizon': 10,
            'inflow_pattern': 'Constant 10 veh/min for t=0-4, then 0',
            'discharge_rates': 'Unlimited (1000 veh/min)'
        },
        'expected_outputs': {
            'total_inflow': 150.0,
            'total_outflow': 150.0,
            'final_queues': 0.0,
            'max_queue_any_link': 0.0,
            'conservation_error': 0.0
        }
    }
    
    # Test Case 2 Data
    test2_data = {
        'description': 'Queue Formation Under Capacity Constraints',
        'network': {
            'num_links': 3,
            'capacities_veh_per_hour': [1800, 2400, 1200]
        },
        'inputs': {
            'time_horizon': 20,
            'inflow_patterns': {
                'link_0': '50 veh/min for t=5-14 (exceeds capacity)',
                'link_1': '30 veh/min for t=5-14 (equals capacity)',
                'link_2': '10 veh/min for t=5-14 (below capacity)'
            },
            'discharge_rates': {
                'link_0': '20 veh/min (bottleneck)',
                'link_1': '30 veh/min (balanced)',
                'link_2': '40 veh/min (uncongested)'
            }
        },
        'expected_outputs': {
            'link_0_max_queue': '>10 vehicles (queue buildup)',
            'link_1_max_queue': '<5 vehicles (minimal queue)',
            'link_2_max_queue': '<1 vehicle (no queue)',
            'total_system_inflow': 900.0,
            'queue_formation_pattern': 'Link 0 accumulates queue due to capacity constraint'
        }
    }
    
    # Export to JSON
    test_data = {
        'model_x_test_cases': {
            'test_case_1': test1_data,
            'test_case_2': test2_data
        },
        'validation_criteria': {
            'flow_conservation': 'total_inflow = total_outflow + remaining_queues',
            'queue_formation': 'queues form when inflow > capacity',
            'non_negativity': 'all queues and flows >= 0',
            'capacity_constraints': 'outflow <= min(available_vehicles, discharge_rate)'
        }
    }
    
    with open('model_x_test_specification.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Test specifications exported to model_x_test_specification.json")
    
    # Also create a simple CSV with key test values
    validation_data = [
        ['Test Case', 'Input Description', 'Expected Output', 'Validation Metric'],
        ['Flow Conservation', '150 veh total inflow, unlimited capacity', '150 veh outflow, 0 queue', 'inflow = outflow + queue'],
        ['Queue Formation', '50 veh/min inflow, 20 veh/min capacity', 'Queue builds up > 10 veh', 'max_queue > 10'],
        ['No Queue Case', '10 veh/min inflow, 40 veh/min capacity', 'No queue formation', 'max_queue < 1'],
    ]
    
    with open('model_x_validation_table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(validation_data)
    
    print("Validation table exported to model_x_validation_table.csv")
    
    return test_data


if __name__ == "__main__":
    print("MODEL X QUEUE DYNAMICS - COMPREHENSIVE TEST AND VALIDATION SUITE")
    print("="*80)
    print("This test suite validates Model X implementation before integration")
    
    # Run the main test suite
    start_time = time.time()
    
    test_results, travel_times = run_all_tests()
    
    if test_results is not None:
        print(f"\n✓ Core tests completed successfully in {time.time() - start_time:.2f} seconds")
        
        # Test with real data if available
        print("\nTesting with real network data...")
        real_results, real_travel_times, real_df = test_with_real_data()
        
        # Performance benchmark
        print("\nRunning performance benchmark...")
        benchmark_df = performance_benchmark()
        
        # Export test specifications
        print("\nExporting test documentation...")
        test_specs = export_test_inputs_outputs()
        
        print("\n" + "="*80)
        print("MODEL X VALIDATION COMPLETE")
        print("="*80)
        print("\nValidation Results:")
        print("✓ Flow conservation equation works correctly")
        print("✓ Queue formation under capacity constraints")
        print("✓ Realistic traffic scenarios handled properly")
        print("✓ Travel time calculations include queueing delays")
        print("✓ Performance benchmarked for various network sizes")
        
        if real_results is not None:
            print("✓ Real network data processed successfully")
        
        print("\nGenerated Files:")
        print("  • model_x_test_results.csv - Detailed link-level results")
        print("  • model_x_benchmark.csv - Performance metrics")
        print("  • model_x_test_specification.json - Test documentation")
        print("  • model_x_validation_table.csv - Validation criteria")
        
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
        print("\n🎉 Model X is validated and ready for integration with Models Y and Z!")
        
    else:
        print(f"\n❌ Test suite failed after {time.time() - start_time:.2f} seconds")
        print("Please review the error messages and fix identified issues.")
        print("\nCommon issues to check:")
        print("  • Array dimensions match (A x T)")
        print("  • Non-negative values for queues and flows")
        print("  • Flow conservation holds at each time step")
        print("  • Capacity constraints are enforced")