# Falsification Protocols GUI - Comprehensive Test Checklist

## GUI Launch Verification

- [ ] GUI window opens without errors
- [ ] Window title displays "APGI Framework-Level Falsification Aggregator (FP-ALL)"
- [ ] Window size is appropriate (800x600 default)
- [ ] Window can be resized (minimum 640x480)
- [ ] Output console displays "APGI Falsification Protocols GUI initialized"
- [ ] Status bar shows "Ready"
- [ ] Progress bar is at 0%

## Tab Navigation

- [ ] "Protocols" tab exists and is accessible
- [ ] "Parameters" tab exists and is accessible
- [ ] Tabs switch correctly when clicked
- [ ] Tab content displays properly

## Protocol Selection (13 Protocols)

### Protocol 1: APGI Agent

- [ ] Button exists and displays "APGI Agent"
- [ ] Button is clickable
- [ ] Tooltip shows "Complete APGI-based active inference agent"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 2: Iowa Gambling

- [ ] Button exists and displays "Iowa Gambling"
- [ ] Button is clickable
- [ ] Tooltip shows "IGT variant with simulated interoceptive costs"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 3: Agent Comparison

- [ ] Button exists and displays "Agent Comparison"
- [ ] Button is clickable
- [ ] Tooltip shows "Run complete agent comparison experiment"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 4: Phase Transition

- [ ] Button exists and displays "Phase Transition"
- [ ] Button is clickable
- [ ] Tooltip shows "Test APGI ignition phase transition signatures"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 5: Evolutionary

- [ ] Button exists and displays "Evolutionary"
- [ ] Button is clickable
- [ ] Tooltip shows "Test APGI emergence under selection pressure"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 6: Network Comparison

- [ ] Button exists and displays "Network Comparison"
- [ ] Button is clickable
- [ ] Tooltip shows "Compare APGI-inspired vs standard architectures"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 7: Mathematical Consistency

- [ ] Button exists and displays "Mathematical Consistency"
- [ ] Button is clickable
- [ ] Tooltip shows "Test mathematical consistency of APGI equations"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 8: Parameter Sensitivity

- [ ] Button exists and displays "Parameter Sensitivity"
- [ ] Button is clickable
- [ ] Tooltip shows "Parameter sensitivity and identifiability analysis"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 9: Neural Signatures

- [ ] Button exists and displays "Neural Signatures"
- [ ] Button is clickable
- [ ] Tooltip shows "Validate P3b and HEP neural signatures"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 10: Bayesian Estimation

- [ ] Button exists and displays "Bayesian Estimation"
- [ ] Button is clickable
- [ ] Tooltip shows "Bayesian parameter recovery analysis"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 11: Liquid Network Dynamics

- [ ] Button exists and displays "Liquid Network Dynamics"
- [ ] Button is clickable
- [ ] Tooltip shows "Liquid network dynamics and echo state analysis"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 12: Cross Species Scaling

- [ ] Button exists and displays "Cross Species Scaling"
- [ ] Button is clickable
- [ ] Tooltip shows "Cross-species allometric scaling and clinical convergence analysis"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

### Protocol 13: Framework Aggregator

- [ ] Button exists and displays "Framework Aggregator"
- [ ] Button is clickable
- [ ] Tooltip shows "Full Framework Falsification Report"
- [ ] Clicking selects the protocol
- [ ] "Selected Protocol" label updates to show selection
- [ ] "Run Selected Protocol" button becomes enabled

## Parameters Tab Testing

### Protocol 1: APGI Agent Parameters

- [ ] Select Protocol 1 from dropdown
- [ ] All 12 parameters display: lr_extero, lr_intero, lr_precision, lr_somatic, n_actions, theta_init, theta_baseline, alpha, tau_S, tau_theta, eta_theta, beta, rho
- [ ] Each parameter has correct type (float/int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 2: Iowa Gambling Parameters

- [ ] Select Protocol 2 from dropdown
- [ ] All 4 parameters display: n_trials, n_decks, interoceptive_cost_weight, learning_rate
- [ ] Each parameter has correct type (float/int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 3: Agent Comparison Parameters

- [ ] Select Protocol 3 from dropdown
- [ ] All 3 parameters display: n_episodes, episode_length, n_agents
- [ ] Each parameter has correct type (int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 4: Phase Transition Parameters

- [ ] Select Protocol 4 from dropdown
- [ ] All 4 parameters display: surprise_range, n_points, tau_S, alpha
- [ ] Each parameter has correct type (str/int/float)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 5: Evolutionary Parameters

- [ ] Select Protocol 5 from dropdown
- [ ] All 4 parameters display: population_size, n_generations, mutation_rate, selection_pressure
- [ ] Each parameter has correct type (int/float)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 6: Network Comparison Parameters

- [ ] Select Protocol 6 from dropdown
- [ ] All 5 parameters display: extero_dim, intero_dim, action_dim, context_dim, n_episodes
- [ ] Each parameter has correct type (int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 7: Mathematical Consistency Parameters

- [ ] Select Protocol 7 from dropdown
- [ ] All 3 parameters display: epsilon, n_samples, tolerance
- [ ] Each parameter has correct type (float/int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 8: Parameter Sensitivity Parameters

- [ ] Select Protocol 8 from dropdown
- [ ] All 3 parameters display: n_samples, n_trials, n_levels
- [ ] Each parameter has correct type (int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 9: Neural Signatures Parameters

- [ ] Select Protocol 9 from dropdown
- [ ] All 5 parameters display: n_participants, sampling_rate, window_size, n_chains, burn_in
- [ ] Each parameter has correct type (int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 10: Bayesian Estimation Parameters

- [ ] Select Protocol 10 from dropdown
- [ ] All 3 parameters display: n_samples, n_chains, burn_in
- [ ] Each parameter has correct type (int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 11: Liquid Network Dynamics Parameters

- [ ] Select Protocol 11 from dropdown
- [ ] All 3 parameters display: spectral_radius, leak_rate, n_units
- [ ] Each parameter has correct type (float/int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 12: Cross Species Scaling Parameters

- [ ] Select Protocol 12 from dropdown
- [ ] All 3 parameters display: spectral_radius, leak_rate, n_nodes
- [ ] Each parameter has correct type (float/int)
- [ ] Each parameter has correct min/max bounds
- [ ] Each parameter has correct default value
- [ ] Parameter values can be modified
- [ ] "Load Defaults" button resets to defaults
- [ ] "Save Parameters" button works

### Protocol 13: Framework Aggregator Parameters

- [ ] Select Protocol 13 from dropdown
- [ ] Message displays "No configurable parameters for this protocol"
- [ ] Parameter controls are empty

## Control Buttons Testing

### Clear Console Button

- [ ] Button exists and is clickable
- [ ] Clicking clears all text from output console
- [ ] Console remains functional after clearing

### Stop Button

- [ ] Button exists and is disabled when no protocol is running
- [ ] Button becomes enabled when a protocol starts running
- [ ] Clicking stops the running protocol
- [ ] Status bar updates to show protocol stopped
- [ ] Button becomes disabled again after stopping

### Run Selected Protocol Button

- [ ] Button exists and is disabled when no protocol is selected
- [ ] Button becomes enabled when a protocol is selected
- [ ] Clicking runs the selected protocol
- [ ] Status bar updates to show protocol running
- [ ] Progress bar updates during execution
- [ ] Output console shows protocol output
- [ ] Protocol completes successfully
- [ ] Status bar shows completion

### Run All Protocols Button

- [ ] Button exists and is always enabled
- [ ] Clicking runs all 13 protocols sequentially
- [ ] Status bar updates to show current protocol
- [ ] Progress bar updates during execution
- [ ] Output console shows output for each protocol
- [ ] All protocols complete successfully
- [ ] Status bar shows all protocols completed

## Output Console Testing

- [ ] Console displays initialization message
- [ ] Console displays protocol selection messages
- [ ] Console displays protocol execution output
- [ ] Console displays error messages if any
- [ ] Console is scrollable
- [ ] Console text is readable
- [ ] Console updates in real-time during execution

## Status Bar Testing

- [ ] Status bar displays "Ready" when idle
- [ ] Status bar updates when protocol is selected
- [ ] Status bar updates when protocol starts running
- [ ] Status bar updates when protocol completes
- [ ] Status bar updates when protocol is stopped
- [ ] Status bar is readable

## Progress Bar Testing

- [ ] Progress bar starts at 0%
- [ ] Progress bar updates during protocol execution
- [ ] Progress bar reaches 100% on completion
- [ ] Progress bar resets when new protocol starts

## Window Management

- [ ] Window can be minimized
- [ ] Window can be maximized
- [ ] Window can be restored
- [ ] Window can be resized
- [ ] Window close button works
- [ ] Proper thread cleanup on close

## File Menu

- [ ] File menu exists
- [ ] Exit command works
- [ ] Exit closes the application cleanly

## Edge Cases

- [ ] Switching protocols while one is running
- [ ] Modifying parameters while protocol is running
- [ ] Closing window while protocol is running
- [ ] Selecting protocol with no parameters
- [ ] Running all protocols with modified parameters

## Summary

Total items to verify: 150+

Test each item systematically and mark as complete when verified.
