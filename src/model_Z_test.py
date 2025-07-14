# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 19:34:15 2025

@author: xzhou
"""

elif pattern == 'realistic':
            # Realistic should have high variation
            demand_values = [d for d in od_demand.values() if d > 0]
            cv = np.std(demand_values) / np.mean(demand_values)
            assert cv > 0.3, f"Realistic pattern should have high variation but got CV={cv:.2f}"
    
    print("✓ TEST CASE 2 PASSED")
    
    return results


def test_case_3_route_choice_logic():
    """Test Case 3: Route Choice Logic"""
    
    print("\n" + "="*80)
    print("TEST CASE 3: ROUTE CHOICE LOGIC")
    print("="*80)
    print("Objective: Verify logit route choice responds correctly to travel times")
    
    # Create test network
    nodes_df, links_df, network_graph = create_test_network()
    model_z = ModelZ(nodes_df, links_df, network_graph, time_horizon=30)
    
    print("\nInput Conditions:")
    print("- Generate OD demand")
    print("- Test route choice with different travel time scenarios")
    print("- Verify logit model sensitivity")
    
    # Generate test demand
    od_demand = model_z.generate_od_demand(peak_hour_factor=1.0, demand_pattern='uniform')
    
    # Test scenarios with different travel times
    scenarios = [
        {
            'name': 'Equal Travel Times',
            'link_times': np.ones(len(links_df)) * 2.0,  # All links 2 minutes
            'expected': 'Equal flow split among paths'
        },
        {
            'name': 'One Slow Link',
            'link_times': np.ones(len(links_df)) * 2.0,
            'slow_link': 0,  # Make first link very slow
            'expected': 'Avoid paths using slow link'
        },
        {
            'name': 'Gradient Travel Times',
            'link_times': np.linspace(1.0, 5.0, len(links_df)),  # Increasing times
            'expected': 'Prefer paths with lower-indexed links'
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Set travel times
        travel_times = scenario['link_times'].copy()
        if 'slow_link' in scenario:
            travel_times[scenario['slow_link']] = 10.0  # Very slow
            print(f"  Made link {scenario['slow_link']} very slow (10 min)")
        
        print(f"  Travel time range: {np.min(travel_times):.1f} - {np.max(travel_times):.1f} minutes")
        
        # Test different logit parameters
        logit_params = [0.05, 0.1, 0.5]  # Low = more sensitive to travel time
        
        for theta in logit_params:
            print(f"\n  Logit parameter θ = {theta}")
            
            # Perform route choice
            path_flows = model_z.route_choice_logit(od_demand, travel_times, theta=theta, verbose=False)
            
            # Analyze results
            total_flow = np.sum(path_flows)
            active_paths = np.sum(path_flows > 0.1)
            flow_concentration = np.max(path_flows) / max(total_flow, 1) if total_flow > 0 else 0
            
            print(f"    Total flow: {total_flow:.0f} veh/h")
            print(f"    Active paths: {active_paths}/{len(path_flows)}")
            print(f"    Flow concentration: {flow_concentration:.3f}")
            
            # For the slow link scenario, check if paths avoid the slow link
            if 'slow_link' in scenario:
                slow_link_id = links_df.iloc[scenario['slow_link']]['link_id']
                flows_using_slow_link = 0
                total_flows_for_affected_ods = 0
                
                for path_id, flow in enumerate(path_flows):
                    if flow > 0.1:  # Active path
                        path_links = model_z.path_links.get(path_id, [])
                        if slow_link_id in path_links:
                            flows_using_slow_link += flow
                        
                        # Count flows for ODs that could use the slow link
                        path_info = model_z.paths.get(path_id, {})
                        od_pair = path_info.get('od_pair')
                        if od_pair and od_pair in od_demand:
                            total_flows_for_affected_ods += flow
                
                slow_link_usage = flows_using_slow_link / max(total_flows_for_affected_ods, 1)
                print(f"    Slow link usage: {slow_link_usage:.3f}")
                
                # Should use slow link less with lower theta (more sensitive)
                if theta == 0.05:
                    assert slow_link_usage < 0.3, f"Should avoid slow link with low theta but usage={slow_link_usage:.3f}"
        
        results[scenario['name']] = {
            'travel_times': travel_times.copy(),
            'path_flows': path_flows.copy()
        }
    
    # Compare sensitivity across scenarios
    print(f"\nSENSITIVITY ANALYSIS:")
    equal_flows = results['Equal Travel Times']['path_flows']
    gradient_flows = results['Gradient Travel Times']['path_flows']
    
    # Flow should be more concentrated with gradient travel times
    equal_concentration = np.max(equal_flows) / max(np.sum(equal_flows), 1)
    gradient_concentration = np.max(gradient_flows) / max(np.sum(gradient_flows), 1)
    
    print(f"Equal times concentration: {equal_concentration:.3f}")
    print(f"Gradient times concentration: {gradient_concentration:.3f}")
    
    assert gradient_concentration > equal_concentration, "Gradient scenario should have higher flow concentration"
    
    print("✓ TEST CASE 3 PASSED")
    
    return results


def test_case_4_temporal_flow_assignment():
    """Test Case 4: Temporal Flow Assignment"""
    
    print("\n" + "="*80)
    print("TEST CASE 4: TEMPORAL FLOW ASSIGNMENT")
    print("="*80)
    print("Objective: Verify temporal distribution and link flow assignment")
    
    # Create test network
    nodes_df, links_df, network_graph = create_test_network()
    model_z = ModelZ(nodes_df, links_df, network_graph, time_horizon=60)
    
    print("\nInput Conditions:")
    print("- Generate path flows from route choice")
    print("- Test different temporal patterns")
    print("- Verify flow conservation and temporal profiles")
    
    # Generate demand and route choice
    od_demand = model_z.generate_od_demand(peak_hour_factor=1.2, demand_pattern='realistic')
    travel_times = np.random.uniform(1.0, 4.0, len(links_df))  # Random travel times
    path_flows = model_z.route_choice_logit(od_demand, travel_times, theta=0.1, verbose=False)
    
    print(f"Generated {np.sum(path_flows):.0f} veh/h of path flows")
    
    # Test different temporal patterns
    temporal_patterns = ['uniform', 'peak', 'double_peak', 'ramp_up']
    results = {}
    
    for pattern in temporal_patterns:
        print(f"\n--- Testing {pattern.upper()} temporal pattern ---")
        
        # Assign flows to links
        link_flows = model_z.assign_flows_to_links(path_flows, temporal_distribution=pattern, verbose=False)
        
        # Analyze temporal pattern
        total_flow_by_time = np.sum(link_flows, axis=0)
        peak_time = np.argmax(total_flow_by_time)
        peak_flow = np.max(total_flow_by_time)
        off_peak_flow = np.min(total_flow_by_time[total_flow_by_time > 0]) if np.any(total_flow_by_time > 0) else 0
        
        # Calculate flow concentration over time
        if np.sum(total_flow_by_time) > 0:
            time_concentration = peak_flow / np.mean(total_flow_by_time[total_flow_by_time > 0])
        else:
            time_concentration = 1.0
        
        print(f"  Peak time: {peak_time} (of {model_z.T})")
        print(f"  Peak flow: {peak_flow * 60:.0f} veh/h")
        print(f"  Off-peak flow: {off_peak_flow * 60:.0f} veh/h")
        print(f"  Time concentration: {time_concentration:.2f}")
        
        # Check flow conservation
        total_link_flow_per_hour = np.sum(link_flows) * 60  # Convert to veh/h
        total_path_flow_per_hour = np.sum(path_flows)
        conservation_ratio = total_link_flow_per_hour / max(total_path_flow_per_hour, 1)
        
        print(f"  Flow conservation: {conservation_ratio:.3f}")
        
        results[pattern] = {
            'link_flows': link_flows.copy(),
            'peak_time': peak_time,
            'time_concentration': time_concentration,
            'conservation_ratio': conservation_ratio
        }
        
        # Pattern-specific tests
        if pattern == 'uniform':
            assert time_concentration < 1.5, f"Uniform should have low concentration but got {time_concentration:.2f}"
        elif pattern == 'peak':
            assert time_concentration > 2.0, f"Peak should have high concentration but got {time_concentration:.2f}"
            assert 15 <= peak_time <= 25, f"Peak should be around time 20 but got {peak_time}"
        elif pattern == 'double_peak':
            # Should have two local maxima
            flow_profile = total_flow_by_time
            peaks = []
            for t in range(1, len(flow_profile) - 1):
                if flow_profile[t] > flow_profile[t-1] and flow_profile[t] > flow_profile[t+1]:
                    if flow_profile[t] > 0.5 * np.max(flow_profile):  # Significant peak
                        peaks.append(t)
            assert len(peaks) >= 1, f"Double peak should have at least 1 significant peak but found {len(peaks)}"
        
        # All patterns should conserve flow
        assert 0.95 <= conservation_ratio <= 1.05, f"Flow conservation violated: {conservation_ratio:.3f}"
    
    # Compare temporal patterns
    print(f"\nTEMPORAL PATTERN COMPARISON:")
    print(f"Pattern      | Peak Time | Concentration | Conservation")
    print(f"-------------|-----------|---------------|-------------")
    for pattern, result in results.items():
        print(f"{pattern:12} | {result['peak_time']:9} | {result['time_concentration']:13.2f} | {result['conservation_ratio']:11.3f}")
    
    print("✓ TEST CASE 4 PASSED")
    
    return results


def test_case_5_iterative_assignment():
    """Test Case 5: Iterative Assignment (MSA)"""
    
    print("\n" + "="*80)
    print("TEST CASE 5: ITERATIVE ASSIGNMENT (METHOD OF SUCCESSIVE AVERAGES)")
    print("="*80)
    print("Objective: Verify convergence of iterative assignment process")
    
    # Create test network
    nodes_df, links_df, network_graph = create_test_network()
    model_z = ModelZ(nodes_df, links_df, network_graph, time_horizon=30)
    
    print("\nInput Conditions:")
    print("- Iterative assignment with MSA step size")
    print("- Travel times depend on link flows (simple VDF)")
    print("- Monitor convergence")
    
    # Generate demand
    od_demand = model_z.generate_od_demand(peak_hour_factor=1.0, demand_pattern='realistic')
    
    # Initialize
    max_iterations = 15
    convergence_threshold = 0.01
    
    # Storage for results
    flow_history = []
    gap_history = []
    
    # Initial travel times (free flow)
    base_travel_times = np.array([links_df.iloc[i]['VDF_fftt'] for i in range(len(links_df))])
    current_flows = np.zeros(len(model_z.paths))
    
    print(f"\nIterative Assignment ({max_iterations} max iterations):")
    print(f"Iteration | Total Flow | Max Flow Change | Gap")
    print(f"----------|------------|-----------------|----")
    
    for iteration in range(max_iterations):
        # Calculate link travel times from current flows
        link_flows_hourly = np.zeros(len(links_df))
        
        # Convert path flows to link flows
        for path_id, flow in enumerate(current_flows):
            if flow > 0:
                path_links = model_z.path_links.get(path_id, [])
                for link_id in path_links:
                    if link_id in model_z.link_id_to_idx:
                        link_idx = model_z.link_id_to_idx[link_id]
                        link_flows_hourly[link_idx] += flow
        
        # Simple VDF: travel_time = free_flow_time * (1 + 0.15 * (flow/capacity)^4)
        travel_times = base_travel_times.copy()
        for i in range(len(links_df)):
            capacity = links_df.iloc[i]['capacity']
            if capacity > 0:
                volume_capacity_ratio = link_flows_hourly[i] / capacity
                travel_times[i] = base_travel_times[i] * (1 + 0.15 * volume_capacity_ratio**4)
        
        # All-or-nothing assignment
        aon_flows = model_z.route_choice_logit(od_demand, travel_times, theta=0.05, verbose=False)
        
        # Calculate convergence measures
        if iteration > 0:
            flow_change = np.abs(aon_flows - current_flows)
            max_flow_change = np.max(flow_change)
            
            # Calculate path costs for gap calculation
            current_path_costs = model_z.calculate_path_costs(travel_times)
            aon_path_costs = model_z.calculate_path_costs(travel_times)
            
            # System travel time (current flows)
            current_system_tt = np.sum(current_flows * current_path_costs)
            
            # System travel time (AON flows)
            aon_system_tt = np.sum(aon_flows * aon_path_costs)
            
            # Relative gap
            if current_system_tt > 0:
                gap = abs(current_system_tt - aon_system_tt) / current_system_tt
            else:
                gap = 0
            
            gap_history.append(gap)
        else:
            max_flow_change = 0
            gap = 1.0
        
        # Update flows using MSA
        current_flows = model_z.update_path_flows_msa(current_flows, aon_flows, iteration)
        
        total_flow = np.sum(current_flows)
        flow_history.append(current_flows.copy())
        
        print(f"{iteration:9d} | {total_flow:10.0f} | {max_flow_change:15.3f} | {gap:7.4f}")
        
        # Check convergence
        if iteration > 2 and gap < convergence_threshold:
            print(f"\nConverged at iteration {iteration} (gap = {gap:.4f})")
            break
    
    # Analyze convergence
    print(f"\nCONVERGENCE ANALYSIS:")
    if len(gap_history) > 0:
        final_gap = gap_history[-1]
        print(f"Final gap: {final_gap:.4f}")
        print(f"Gap reduction: {gap_history[0]:.4f} → {final_gap:.4f}")
        
        # Check convergence trend
        if len(gap_history) >= 5:
            recent_gaps = gap_history[-5:]
            gap_trend = np.mean(np.diff(recent_gaps))  # Should be negative (decreasing)
            print(f"Recent gap trend: {gap_trend:.6f} (should be negative)")
        
        converged = final_gap < convergence_threshold
        print(f"Converged: {'✓' if converged else '❌'}")
    else:
        converged = False
    
    # Final flow analysis
    print(f"\nFINAL FLOW DISTRIBUTION:")
    final_flows = current_flows
    active_paths = np.sum(final_flows > 0.1)
    flow_concentration = np.max(final_flows) / max(np.sum(final_flows), 1)
    
    print(f"Active paths: {active_paths}/{len(final_flows)}")
    print(f"Flow concentration: {flow_concentration:.3f}")
    print(f"Total assigned flow: {np.sum(final_flows):.0f} veh/h")
    
    # Test assertions
    assert np.sum(final_flows) > 0, "Should have positive flows"
    assert active_paths > 0, "Should have active paths"
    
    if len(gap_history) > 0:
        assert gap_history[-1] < 0.1, f"Gap should be reasonable but got {gap_history[-1]:.4f}"
    
    print("✓ TEST CASE 5 PASSED")
    
    return {
        'flow_history': flow_history,
        'gap_history': gap_history,
        'final_flows': final_flows,
        'converged': converged
    }


def visualize_model_z_results(test_results: Dict):
    """Visualize Model Z test results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Z Vehicle Routing Test Results', fontsize=16, fontweight='bold')
    
    # Test Case 2: OD Demand Patterns
    if 'od_demand' in test_results:
        demand_results = test_results['od_demand']
        patterns = list(demand_results.keys())
        total_demands = [demand_results[p]['total_demand'] for p in patterns]
        
        axes[0, 0].bar(patterns, total_demands, alpha=0.7, color=['blue', 'green', 'orange'])
        axes[0, 0].set_title('OD Demand by Pattern')
        axes[0, 0].set_ylabel('Total Demand (veh/h)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
    
    # Test Case 4: Temporal Patterns
    if 'temporal_assignment' in test_results:
        temporal_results = test_results['temporal_assignment']
        
        # Show flow profiles for different patterns
        colors = ['blue', 'red', 'green', 'orange']
        for i, (pattern, result) in enumerate(temporal_results.items()):
            if i < 4:  # Limit to 4 patterns
                link_flows = result['link_flows']
                total_flow_by_time = np.sum(link_flows, axis=0)
                time_steps = range(len(total_flow_by_time))
                
                axes[0, 1].plot(time_steps, total_flow_by_time * 60, 
                               label=pattern, color=colors[i], linewidth=2)
        
        axes[0, 1].set_title('Temporal Flow Patterns')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('System Flow (veh/h)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Test Case 5: Convergence
    if 'iterative_assignment' in test_results:
        iter_results = test_results['iterative_assignment']
        gap_history = iter_results.get('gap_history', [])
        
        if len(gap_history) > 0:
            iterations = range(1, len(gap_history) + 1)
            axes[0, 2].semilogy(iterations, gap_history, 'b-o', linewidth=2, markersize=4)
            axes[0, 2].set_title('Convergence: Relative Gap')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Relative Gap (log scale)')
            axes[0, 2].grid(True, alpha=0.3)
    
    # Route Choice Sensitivity
    if 'route_choice' in test_results:
        route_results = test_results['route_choice']
        
        # Compare flow concentration across scenarios
        scenarios = list(route_results.keys())
        concentrations = []
        
        for scenario in scenarios:
            flows = route_results[scenario]['path_flows']
            total_flow = np.sum(flows)
            concentration = np.max(flows) / max(total_flow, 1)
            concentrations.append(concentration)
        
        axes[1, 0].bar(scenarios, concentrations, alpha=0.7, color=['lightblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Flow Concentration by Scenario')
        axes[1, 0].set_ylabel('Flow Concentration')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Network Connectivity (if model_z available)
    try:
        if hasattr(test_results, 'model_z'):
            model_z = test_results['model_z']
            
            # Create network plot
            pos = nx.spring_layout(model_z.network_graph)
            nx.draw_networkx_nodes(model_z.network_graph, pos, ax=axes[1, 1], 
                                 node_color='lightblue', node_size=300)
            nx.draw_networkx_edges(model_z.network_graph, pos, ax=axes[1, 1], 
                                 alpha=0.5, width=1)
            nx.draw_networkx_labels(model_z.network_graph, pos, ax=axes[1, 1], 
                                  font_size=8)
            axes[1, 1].set_title('Network Topology')
            axes[1, 1].axis('off')
    except:
        axes[1, 1].text(0.5, 0.5, 'Network topology\nnot available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Network Topology')
    
    # Performance Summary
    summary_data = {
        'Total Paths': 0,
        'OD Pairs': 0,
        'Connectivity': 0,
        'Convergence': 0
    }
    
    try:
        if 'connectivity' in test_results:
            conn_results = test_results['connectivity']
            summary_data['Total Paths'] = len(conn_results.paths) if hasattr(conn_results, 'paths') else 0
            summary_data['OD Pairs'] = len(conn_results.od_pairs) if hasattr(conn_results, 'od_pairs') else 0
        
        if 'iterative_assignment' in test_results:
            iter_results = test_results['iterative_assignment']
            summary_data['Convergence'] = 1 if iter_results.get('converged', False) else 0
    except:
        pass
    
    metrics = list(summary_data.keys())
    values = list(summary_data.values())
    
    axes[1, 2].bar(metrics, values, alpha=0.7, color=['purple', 'brown', 'pink', 'gray'])
    axes[1, 2].set_title('Model Z Performance Summary')
    axes[1, 2].set_ylabel('Count/Status')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def export_model_z_results(test_results: Dict, output_prefix: str = "model_z_test"):
    """Export Model Z test results to files"""
    
    # Export connectivity results
    if 'connectivity' in test_results:
        model_z = test_results['connectivity']
        
        # Path information
        path_records = []
        for path_id, path_info in model_z.paths.items():
            record = {
                'path_id': path_id,
                'origin': path_info['origin'],
                'destination': path_info['destination'],
                'path_links': ','.join(map(str, path_info['path_links'])),
                'travel_time': path_info['travel_time'],
                'distance': path_info['distance']
            }
            path_records.append(record)
        
        path_df = pd.DataFrame(path_records)
        path_df.to_csv(f"{output_prefix}_paths.csv", index=False)
    
    # Export OD demand results
    if 'od_demand' in test_results:
        demand_summary = []
        for pattern, stats in test_results['od_demand'].items():
            record = {'pattern': pattern}
            record.update(stats)
            demand_summary.append(record)
        
        demand_df = pd.DataFrame(demand_summary)
        demand_df.to_csv(f"{output_prefix}_demand_patterns.csv", index=False)
    
    # Export convergence results
    if 'iterative_assignment' in test_results:
        iter_results = test_results['iterative_assignment']
        gap_history = iter_results.get('gap_history', [])
        
        if len(gap_history) > 0:
            conv_df = pd.DataFrame({
                'iteration': range(1, len(gap_history) + 1),
                'relative_gap': gap_history
            })
            conv_df.to_csv(f"{output_prefix}_convergence.csv", index=False)
    
    print(f"Model Z test results exported to {output_prefix}_*.csv files")


def run_all_model_z_tests():
    """Run all Model Z tests"""
    
    print("MODEL Z VEHICLE ROUTING - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing Model Z components individually before integration")
    
    test_results = {}
    
    try:
        # Test Case 1: Network Connectivity
        model_z = test_case_1_network_connectivity()
        test_results['connectivity'] = model_z
        
        # Test Case 2: OD Demand Generation
        demand_results = test_case_2_od_demand_generation()
        test_results['od_demand'] = demand_results
        
        # Test Case 3: Route Choice Logic
        route_results = test_case_3_route_choice_logic()
        test_results['route_choice'] = route_results
        
        # Test Case 4: Temporal Flow Assignment
        temporal_results = test_case_4_temporal_flow_assignment()
        test_results['temporal_assignment'] = temporal_results
        
        # Test Case 5: Iterative Assignment
        iter_results = test_case_5_iterative_assignment()
        test_results['iterative_assignment'] = iter_results
        
        print("\n" + "="*80)
        print("ALL MODEL Z TESTS PASSED SUCCESSFULLY!")
        print("="*80)
        print("Model Z is working correctly and ready for integration")
        
        # Export results
        export_model_z_results(test_results)
        
        # Visualize results
        print("\nGenerating visualization...")
        visualize_model_z_results(test_results)
        
        return test_results
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return None
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_with_real_data(node_file='node.csv', link_file='link.csv'):
    """Test Model Z with real network data"""
    
    print("\n" + "="*80)
    print("MODEL Z TEST WITH REAL NETWORK DATA")
    print("="*80)
    
    try:
        # Load real network data
        nodes_df = pd.read_csv(node_file)
        links_df = pd.read_csv(link_file)
        print(f"Loaded {len(nodes_df)} nodes and {len(links_df)} links")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for _, node in nodes_df.iterrows():
            G.add_node(node['node_id'], 
                      x=node.get('x_coord', 0), 
                      y=node.get('y_coord', 0),
                      zone_id=node.get('zone_id', 0))
        
        # Add edges
        for _, link in links_df.iterrows():
            G.add_edge(link['from_node_id'], link['to_node_id'],
                      link_id=link['link_id'],
                      length=link['length'],
                      capacity=link['capacity'],
                      free_speed=link.get('free_speed', 50),
                      lanes=link.get('lanes', 1),
                      fftt=link.get('VDF_fftt', link['length'] / max(link.get('free_speed', 50), 1) * 60))
        
        # Initialize Model Z
        model_z = ModelZ(nodes_df, links_df, G, time_horizon=60)
        
        # Generate demand and test basic functionality
        od_demand = model_z.generate_od_demand(peak_hour_factor=0.8, demand_pattern='gravity')
        
        # Quick route choice test
        travel_times = np.array([link.get('VDF_fftt', 2.0) for _, link in links_df.iterrows()])
        path_flows = model_z.route_choice_logit(od_demand, travel_times, theta=0.1, verbose=False)
        
        # Assign to links
        link_flows = model_z.assign_flows_to_links(path_flows, temporal_distribution='peak', verbose=False)
        
        # Analyze results
        print(f"\nREAL NETWORK RESULTS:")
        print(f"Network size: {len(nodes_df)} nodes, {len(links_df)} links")
        print(f"Zones identified: {model_z.num_zones}")
        print(f"OD pairs: {len(model_z.od_pairs)}")
        print(f"#!/usr/bin/env python3
"""
Model Z (Vehicle Routing) Test Suite
Testing space-time-phase vehicle routing and flow assignment individually
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional
import time
import json
import csv
from collections import defaultdict
import heapq


class ModelZ:
    """Model Z: Vehicle Routing - Standalone Testing Version"""
    
    def __init__(self, nodes_df: pd.DataFrame, links_df: pd.DataFrame, 
                 network_graph: nx.DiGraph, time_horizon: int = 60, num_phases: int = 4):
        self.nodes_df = nodes_df
        self.links_df = links_df
        self.network_graph = network_graph
        self.T = time_horizon
        self.P = num_phases
        
        # Network dimensions
        self.A = len(links_df)  # Number of links
        self.N = len(nodes_df)  # Number of nodes
        
        # Identify zones (nodes with zone_id > 0)
        if 'zone_id' in nodes_df.columns:
            self.zones = nodes_df[nodes_df['zone_id'] > 0]['node_id'].tolist()
        else:
            # If no zone_id, use first few nodes as zones
            self.zones = nodes_df['node_id'].head(4).tolist()
        
        self.num_zones = len(self.zones)
        
        # Create mappings
        self.link_id_to_idx = {row['link_id']: idx for idx, row in links_df.iterrows()}
        self.node_id_to_idx = {row['node_id']: idx for idx, row in nodes_df.iterrows()}
        
        # Generate OD pairs and find paths
        self._generate_od_pairs()
        self._find_shortest_paths()
        
        # Initialize path-based variables
        self.num_paths = len(self.path_links)
        self.path_flows = np.zeros((self.num_paths, self.T))
        
        print(f"Model Z initialized:")
        print(f"  {self.A} links, {self.N} nodes, {self.num_zones} zones")
        print(f"  {len(self.od_pairs)} OD pairs, {self.num_paths} paths found")
        print(f"  Time horizon: {self.T} steps, {self.P} phases")
    
    def _generate_od_pairs(self):
        """Generate OD pairs from zones"""
        self.od_pairs = []
        for origin in self.zones:
            for destination in self.zones:
                if origin != destination:
                    self.od_pairs.append((origin, destination))
        
        print(f"Generated {len(self.od_pairs)} OD pairs from {len(self.zones)} zones")
    
    def _find_shortest_paths(self, max_paths_per_od: int = 3):
        """Find multiple shortest paths for each OD pair"""
        self.paths = {}
        self.path_links = {}
        self.path_travel_times = {}
        self.path_distances = {}
        
        path_id = 0
        
        for origin, destination in self.od_pairs:
            od_paths = []
            od_path_links = []
            od_travel_times = []
            od_distances = []
            
            try:
                # Find k shortest paths using different approaches
                paths_found = 0
                
                # Method 1: Shortest path by travel time
                try:
                    path = nx.shortest_path(self.network_graph, origin, destination, weight='fftt')
                    links, travel_time, distance = self._path_to_links(path)
                    
                    od_paths.append(path)
                    od_path_links.append(links)
                    od_travel_times.append(travel_time)
                    od_distances.append(distance)
                    paths_found += 1
                except:
                    pass
                
                # Method 2: Shortest path by distance (if different)
                if paths_found < max_paths_per_od:
                    try:
                        path = nx.shortest_path(self.network_graph, origin, destination, weight='length')
                        links, travel_time, distance = self._path_to_links(path)
                        
                        # Only add if different from previous paths
                        if links not in od_path_links:
                            od_paths.append(path)
                            od_path_links.append(links)
                            od_travel_times.append(travel_time)
                            od_distances.append(distance)
                            paths_found += 1
                    except:
                        pass
                
                # Method 3: Alternative path by removing highest capacity link
                if paths_found < max_paths_per_od and len(od_path_links) > 0:
                    try:
                        # Remove the highest capacity link from the first path
                        first_path_links = od_path_links[0]
                        if len(first_path_links) > 1:
                            # Find link with highest capacity in first path
                            capacities = [self.network_graph.edges[self._get_edge_from_link(link)].get('capacity', 1000) 
                                        for link in first_path_links]
                            max_cap_idx = np.argmax(capacities)
                            link_to_remove = first_path_links[max_cap_idx]
                            
                            # Get the edge corresponding to this link
                            edge_to_remove = self._get_edge_from_link(link_to_remove)
                            
                            # Temporarily remove the edge
                            if edge_to_remove in self.network_graph.edges:
                                self.network_graph.remove_edge(*edge_to_remove)
                                
                                # Find alternative path
                                alt_path = nx.shortest_path(self.network_graph, origin, destination, weight='fftt')
                                links, travel_time, distance = self._path_to_links(alt_path)
                                
                                if links not in od_path_links:
                                    od_paths.append(alt_path)
                                    od_path_links.append(links)
                                    od_travel_times.append(travel_time)
                                    od_distances.append(distance)
                                    paths_found += 1
                                
                                # Restore the edge
                                link_data = self.links_df[self.links_df['link_id'] == link_to_remove].iloc[0]
                                self.network_graph.add_edge(
                                    link_data['from_node_id'], 
                                    link_data['to_node_id'],
                                    **{col: link_data[col] for col in link_data.index if col not in ['from_node_id', 'to_node_id']}
                                )
                    except:
                        pass
                
                # Store all paths found for this OD pair
                for i in range(len(od_paths)):
                    self.paths[path_id] = {
                        'origin': origin,
                        'destination': destination,
                        'path_nodes': od_paths[i],
                        'path_links': od_path_links[i],
                        'travel_time': od_travel_times[i],
                        'distance': od_distances[i],
                        'od_pair': (origin, destination)
                    }
                    
                    self.path_links[path_id] = od_path_links[i]
                    self.path_travel_times[path_id] = od_travel_times[i]
                    self.path_distances[path_id] = od_distances[i]
                    
                    path_id += 1
                
            except nx.NetworkXNoPath:
                print(f"No path found from {origin} to {destination}")
        
        print(f"Found {len(self.paths)} total paths for {len(self.od_pairs)} OD pairs")
    
    def _get_edge_from_link(self, link_id):
        """Get the edge (from_node, to_node) for a given link_id"""
        link_data = self.links_df[self.links_df['link_id'] == link_id].iloc[0]
        return (link_data['from_node_id'], link_data['to_node_id'])
    
    def _path_to_links(self, path):
        """Convert node path to link sequence and calculate metrics"""
        links = []
        total_travel_time = 0
        total_distance = 0
        
        for i in range(len(path) - 1):
            from_node = path[i]
            to_node = path[i + 1]
            
            # Find the link
            edge_data = self.network_graph[from_node][to_node]
            link_id = edge_data['link_id']
            
            links.append(link_id)
            total_travel_time += edge_data.get('fftt', 1.0)
            total_distance += edge_data.get('length', 1000)
        
        return links, total_travel_time, total_distance
    
    def generate_od_demand(self, peak_hour_factor: float = 1.0, 
                          demand_pattern: str = 'uniform') -> Dict:
        """Generate synthetic OD demand matrix"""
        
        od_demand = {}
        
        if demand_pattern == 'uniform':
            # Uniform demand across all OD pairs
            base_demand = 100  # vehicles per hour
            for origin, destination in self.od_pairs:
                od_demand[(origin, destination)] = base_demand * peak_hour_factor
                
        elif demand_pattern == 'gravity':
            # Gravity model based on network distance
            for origin, destination in self.od_pairs:
                try:
                    # Find shortest path distance
                    path_length = nx.shortest_path_length(
                        self.network_graph, origin, destination, weight='length'
                    )
                    
                    # Gravity model: demand inversely proportional to distance
                    base_demand = max(50, 5000 / (1 + path_length / 1000))
                    demand = base_demand * peak_hour_factor * np.random.uniform(0.7, 1.3)
                    od_demand[(origin, destination)] = demand
                    
                except nx.NetworkXNoPath:
                    od_demand[(origin, destination)] = 0
                    
        elif demand_pattern == 'realistic':
            # Realistic demand with major flows and minor flows
            major_pairs = self.od_pairs[:len(self.od_pairs)//3]  # First third are major
            minor_pairs = self.od_pairs[len(self.od_pairs)//3:]  # Rest are minor
            
            for origin, destination in major_pairs:
                base_demand = np.random.uniform(200, 500)  # High demand
                od_demand[(origin, destination)] = base_demand * peak_hour_factor
            
            for origin, destination in minor_pairs:
                base_demand = np.random.uniform(20, 100)   # Low demand
                od_demand[(origin, destination)] = base_demand * peak_hour_factor
        
        total_demand = sum(od_demand.values())
        print(f"Generated {demand_pattern} OD demand: {total_demand:.0f} total veh/h")
        
        return od_demand
    
    def route_choice_logit(self, od_demand: Dict, travel_times: np.ndarray, 
                          theta: float = 0.1, verbose: bool = False) -> np.ndarray:
        """
        Perform route choice using multinomial logit model
        
        Args:
            od_demand: Dictionary of OD demand
            travel_times: Current link travel times
            theta: Logit parameter (lower = more sensitive to travel time)
            verbose: Print detailed route choice information
        """
        if verbose:
            print("\n" + "="*60)
            print("ROUTE CHOICE - LOGIT MODEL")
            print("="*60)
        
        path_flows = np.zeros(len(self.paths))
        
        # Group paths by OD pair
        od_paths = defaultdict(list)
        for path_id, path_info in self.paths.items():
            od_pair = path_info['od_pair']
            od_paths[od_pair].append(path_id)
        
        for od_pair, demand in od_demand.items():
            if demand <= 0:
                continue
                
            if od_pair not in od_paths:
                if verbose:
                    print(f"No paths found for OD {od_pair}")
                continue
            
            path_ids = od_paths[od_pair]
            
            # Calculate path costs (travel times)
            path_costs = []
            for path_id in path_ids:
                path_links = self.path_links[path_id]
                total_cost = 0
                
                for link_id in path_links:
                    if link_id in self.link_id_to_idx:
                        link_idx = self.link_id_to_idx[link_id]
                        total_cost += travel_times[link_idx]
                    else:
                        total_cost += 1.0  # Default travel time
                
                path_costs.append(total_cost)
            
            path_costs = np.array(path_costs)
            
            # Logit probabilities
            if len(path_costs) > 1:
                exp_costs = np.exp(-path_costs / theta)
                probabilities = exp_costs / np.sum(exp_costs)
            else:
                probabilities = np.array([1.0])
            
            # Assign flows
            for i, path_id in enumerate(path_ids):
                path_flows[path_id] = demand * probabilities[i]
            
            if verbose and demand > 100:  # Show details for major flows
                print(f"\nOD {od_pair} (demand: {demand:.0f}):")
                for i, path_id in enumerate(path_ids):
                    print(f"  Path {path_id}: cost={path_costs[i]:.2f}, "
                          f"prob={probabilities[i]:.3f}, flow={path_flows[path_id]:.1f}")
        
        if verbose:
            total_assigned = np.sum(path_flows)
            total_demand = sum(od_demand.values())
            print(f"\nTotal demand: {total_demand:.0f}, Total assigned: {total_assigned:.0f}")
        
        return path_flows
    
    def assign_flows_to_links(self, path_flows: np.ndarray, 
                             temporal_distribution: str = 'peak',
                             verbose: bool = False) -> np.ndarray:
        """
        Convert path flows to link flows with temporal distribution
        
        Args:
            path_flows: Flow on each path [veh/h]
            temporal_distribution: Type of temporal pattern
            verbose: Print detailed assignment information
        """
        if verbose:
            print("\n" + "="*60)
            print("LINK FLOW ASSIGNMENT")
            print("="*60)
        
        # Initialize link flows
        link_flows = np.zeros((self.A, self.T))
        
        # Create temporal demand pattern
        time_factors = self._create_temporal_pattern(temporal_distribution)
        
        # Convert path flows to link flows
        for path_id, hourly_flow in enumerate(path_flows):
            if hourly_flow <= 0:
                continue
            
            path_links = self.path_links.get(path_id, [])
            
            # Distribute flow over time
            for t in range(self.T):
                time_flow = hourly_flow * time_factors[t] / 60  # Convert to veh/min
                
                # Add to each link in the path
                for link_id in path_links:
                    if link_id in self.link_id_to_idx:
                        link_idx = self.link_id_to_idx[link_id]
                        link_flows[link_idx, t] += time_flow
        
        if verbose:
            total_link_flow = np.sum(link_flows) * 60  # Convert back to veh/h
            total_path_flow = np.sum(path_flows)
            
            print(f"Total path flow: {total_path_flow:.0f} veh/h")
            print(f"Total link flow: {total_link_flow:.0f} veh/h")
            print(f"Flow conservation ratio: {total_link_flow / max(total_path_flow, 1):.3f}")
            
            # Show top loaded links
            max_link_flows = np.max(link_flows, axis=1)
            top_links = np.argsort(max_link_flows)[-5:][::-1]
            
            print(f"\nTop 5 loaded links:")
            for rank, link_idx in enumerate(top_links):
                if max_link_flows[link_idx] > 0:
                    link_id = self.links_df.iloc[link_idx]['link_id']
                    max_flow = max_link_flows[link_idx] * 60  # Convert to veh/h
                    print(f"  {rank+1}. Link {link_id}: {max_flow:.0f} veh/h")
        
        return link_flows
    
    def _create_temporal_pattern(self, pattern_type: str) -> np.ndarray:
        """Create temporal distribution pattern"""
        
        if pattern_type == 'uniform':
            # Uniform distribution over time
            return np.ones(self.T) / self.T
            
        elif pattern_type == 'peak':
            # Morning peak pattern (bell curve)
            peak_time = self.T // 3  # Peak at 1/3 of horizon
            sigma = self.T // 8      # Standard deviation
            
            times = np.arange(self.T)
            pattern = np.exp(-0.5 * ((times - peak_time) / sigma)**2)
            return pattern / np.sum(pattern)
            
        elif pattern_type == 'double_peak':
            # Morning and evening peaks
            peak1_time = self.T // 4
            peak2_time = 3 * self.T // 4
            sigma = self.T // 10
            
            times = np.arange(self.T)
            pattern1 = np.exp(-0.5 * ((times - peak1_time) / sigma)**2)
            pattern2 = np.exp(-0.5 * ((times - peak2_time) / sigma)**2)
            pattern = 0.6 * pattern1 + 0.4 * pattern2  # Morning peak stronger
            return pattern / np.sum(pattern)
            
        elif pattern_type == 'ramp_up':
            # Gradually increasing demand
            pattern = np.linspace(0.5, 2.0, self.T)
            return pattern / np.sum(pattern)
        
        else:
            # Default to uniform
            return np.ones(self.T) / self.T
    
    def calculate_path_costs(self, link_travel_times: np.ndarray, 
                           time_step: int = None) -> np.ndarray:
        """Calculate path travel costs from link travel times"""
        
        if time_step is None:
            # Use average travel times
            avg_travel_times = np.mean(link_travel_times, axis=1) if link_travel_times.ndim > 1 else link_travel_times
        else:
            avg_travel_times = link_travel_times[:, time_step] if link_travel_times.ndim > 1 else link_travel_times
        
        path_costs = np.zeros(len(self.paths))
        
        for path_id, path_links in self.path_links.items():
            total_cost = 0
            for link_id in path_links:
                if link_id in self.link_id_to_idx:
                    link_idx = self.link_id_to_idx[link_id]
                    total_cost += avg_travel_times[link_idx]
                else:
                    total_cost += 1.0  # Default
            
            path_costs[path_id] = total_cost
        
        return path_costs
    
    def update_path_flows_msa(self, current_flows: np.ndarray, new_flows: np.ndarray, 
                             iteration: int) -> np.ndarray:
        """Update path flows using Method of Successive Averages"""
        step_size = 1.0 / (iteration + 1)
        return (1 - step_size) * current_flows + step_size * new_flows


def create_test_network() -> Tuple[pd.DataFrame, pd.DataFrame, nx.DiGraph]:
    """Create a test network for Model Z testing"""
    
    # Create nodes
    nodes_data = {
        'node_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'zone_id': [1, 2, 3, 4, 0, 0, 0, 0],  # First 4 are zones
        'x_coord': [0, 300, 600, 300, 150, 450, 150, 450],
        'y_coord': [0, 0, 0, 300, 150, 150, 450, 450]
    }
    nodes_df = pd.DataFrame(nodes_data)
    
    # Create links (grid-like network)
    links_data = {
        'link_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'from_node_id': [1, 2, 3, 4, 1, 2, 5, 6, 5, 6, 7, 8],
        'to_node_id': [2, 3, 4, 1, 5, 6, 7, 8, 3, 4, 3, 4],
        'length': [300, 300, 300, 300, 200, 200, 200, 200, 350, 350, 350, 350],
        'capacity': [1800, 2000, 1600, 1800, 1200, 1400, 1000, 1200, 1600, 1800, 1400, 1600],
        'free_speed': [50, 55, 45, 50, 40, 45, 35, 40, 45, 50, 40, 45],
        'lanes': [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
        'VDF_fftt': [3.6, 3.3, 4.0, 3.6, 3.0, 2.7, 3.4, 3.0, 4.7, 4.2, 5.3, 4.7]
    }
    links_df = pd.DataFrame(links_data)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for _, node in nodes_df.iterrows():
        G.add_node(node['node_id'], 
                  x=node['x_coord'], 
                  y=node['y_coord'],
                  zone_id=node['zone_id'])
    
    # Add edges
    for _, link in links_df.iterrows():
        G.add_edge(link['from_node_id'], link['to_node_id'],
                  link_id=link['link_id'],
                  length=link['length'],
                  capacity=link['capacity'],
                  free_speed=link['free_speed'],
                  lanes=link['lanes'],
                  fftt=link['VDF_fftt'])
    
    return nodes_df, links_df, G


def test_case_1_network_connectivity():
    """Test Case 1: Network Connectivity and Path Finding"""
    
    print("\n" + "="*80)
    print("TEST CASE 1: NETWORK CONNECTIVITY AND PATH FINDING")
    print("="*80)
    print("Objective: Verify that paths can be found between all zone pairs")
    
    # Create test network
    nodes_df, links_df, network_graph = create_test_network()
    model_z = ModelZ(nodes_df, links_df, network_graph, time_horizon=30)
    
    print("\nInput Conditions:")
    print(f"- {len(nodes_df)} nodes ({model_z.num_zones} zones)")
    print(f"- {len(links_df)} links")
    print(f"- Grid-like network topology")
    
    print("\nExpected Outputs:")
    print("- Paths found for all zone pairs")
    print("- Multiple alternative paths where possible")
    print("- Reasonable path lengths and travel times")
    
    # Analyze path finding results
    print(f"\nPATH ANALYSIS:")
    print(f"Total OD pairs: {len(model_z.od_pairs)}")
    print(f"Paths found: {len(model_z.paths)}")
    print(f"Average paths per OD: {len(model_z.paths) / max(len(model_z.od_pairs), 1):.1f}")
    
    # Check connectivity
    connected_pairs = 0
    for od_pair in model_z.od_pairs:
        od_paths = [p for p in model_z.paths.values() if p['od_pair'] == od_pair]
        if len(od_paths) > 0:
            connected_pairs += 1
    
    connectivity_rate = connected_pairs / len(model_z.od_pairs)
    print(f"Connectivity rate: {connectivity_rate:.1%}")
    
    # Analyze path characteristics
    if len(model_z.paths) > 0:
        travel_times = [p['travel_time'] for p in model_z.paths.values()]
        distances = [p['distance'] for p in model_z.paths.values()]
        
        print(f"\nPath Characteristics:")
        print(f"Travel time range: {min(travel_times):.1f} - {max(travel_times):.1f} minutes")
        print(f"Distance range: {min(distances):.0f} - {max(distances):.0f} units")
        print(f"Average travel time: {np.mean(travel_times):.1f} minutes")
        print(f"Average distance: {np.mean(distances):.0f} units")
        
        # Show sample paths
        print(f"\nSample Paths:")
        for i, (path_id, path_info) in enumerate(list(model_z.paths.items())[:5]):
            od_pair = path_info['od_pair']
            links = path_info['path_links']
            travel_time = path_info['travel_time']
            print(f"  Path {path_id}: {od_pair[0]}→{od_pair[1]}, "
                  f"Links: {links}, Time: {travel_time:.1f} min")
    
    # Test assertions
    assert connectivity_rate >= 0.8, f"Connectivity rate too low: {connectivity_rate:.1%}"
    assert len(model_z.paths) >= len(model_z.od_pairs), "Should have at least one path per OD pair"
    
    print("✓ TEST CASE 1 PASSED")
    
    return model_z


def test_case_2_od_demand_generation():
    """Test Case 2: OD Demand Generation"""
    
    print("\n" + "="*80)
    print("TEST CASE 2: OD DEMAND GENERATION")
    print("="*80)
    print("Objective: Verify OD demand generation with different patterns")
    
    # Create test network
    nodes_df, links_df, network_graph = create_test_network()
    model_z = ModelZ(nodes_df, links_df, network_graph, time_horizon=30)
    
    print("\nInput Conditions:")
    print("- Test different demand patterns: uniform, gravity, realistic")
    print("- Verify demand properties and distributions")
    
    # Test different demand patterns
    demand_patterns = ['uniform', 'gravity', 'realistic']
    results = {}
    
    for pattern in demand_patterns:
        print(f"\n--- Testing {pattern.upper()} demand pattern ---")
        
        od_demand = model_z.generate_od_demand(peak_hour_factor=1.5, demand_pattern=pattern)
        
        # Analyze demand
        total_demand = sum(od_demand.values())
        non_zero_pairs = sum(1 for d in od_demand.values() if d > 0)
        avg_demand = total_demand / max(non_zero_pairs, 1)
        max_demand = max(od_demand.values()) if od_demand else 0
        min_demand = min(d for d in od_demand.values() if d > 0) if od_demand else 0
        
        results[pattern] = {
            'total_demand': total_demand,
            'non_zero_pairs': non_zero_pairs,
            'avg_demand': avg_demand,
            'max_demand': max_demand,
            'min_demand': min_demand
        }
        
        print(f"  Total demand: {total_demand:.0f} veh/h")
        print(f"  Non-zero OD pairs: {non_zero_pairs}/{len(model_z.od_pairs)}")
        print(f"  Average demand: {avg_demand:.1f} veh/h")
        print(f"  Demand range: {min_demand:.0f} - {max_demand:.0f} veh/h")
        
        # Show top demand pairs
        sorted_demand = sorted(od_demand.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3 demand pairs:")
        for i, ((o, d), demand) in enumerate(sorted_demand[:3]):
            print(f"    {i+1}. {o}→{d}: {demand:.0f} veh/h")
    
    print(f"\nCOMPARISON ACROSS PATTERNS:")
    for pattern, stats in results.items():
        print(f"{pattern:>10}: Total={stats['total_demand']:6.0f}, "
              f"Avg={stats['avg_demand']:5.1f}, Range={stats['min_demand']:3.0f}-{stats['max_demand']:3.0f}")
    
    # Test assertions
    for pattern, stats in results.items():
        assert stats['total_demand'] > 0, f"{pattern} pattern should generate positive demand"
        assert stats['non_zero_pairs'] > 0, f"{pattern} pattern should have non-zero OD pairs"
        
        if pattern == 'uniform':
            # Uniform should have similar demands
            demand_values = [d for d in od_demand.values() if d > 0]
            cv = np.std(demand_values) / np.mean(demand_values)  # Coefficient of variation
            assert cv < 0.5, f"Uniform pattern should have low variation but got CV={cv:.2f}"
        
        elif pattern == 'realistic':
            # Realistic should have high variation
            demand_values = [d for d in od_demand.values() if d >