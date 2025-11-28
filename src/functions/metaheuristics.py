"""Funções otimizadas para avaliação do problema SSP-NPM."""

import numpy as np
from numba import njit


@njit
def check_machine_eligibility(job_id, machine_id, magazines_capacities, tools_per_job):
    """Verifica se uma tarefa pode ser alocada a uma máquina."""
    return tools_per_job[job_id] <= magazines_capacities[machine_id]


@njit
def calculate_makespan_for_machine(machine_jobs, tools_requirements_matrix, magazine_capacity, 
                                   tool_change_cost, job_costs):
    """Calcula o makespan de uma máquina considerando trocas de ferramentas."""
    if len(machine_jobs) == 0:
        return 0
    
    num_tools = tools_requirements_matrix.shape[0]
    magazine = np.zeros(num_tools, dtype=np.int64)
    elapsed_time = 0
    
    start_switch_index = len(machine_jobs)
    for i, job_id in enumerate(machine_jobs):
        job_tools = tools_requirements_matrix[:, job_id]
        temp_magazine = np.logical_or(magazine, job_tools).astype(np.int64)
        
        if np.sum(temp_magazine) > magazine_capacity:
            start_switch_index = i
            break
        else:
            magazine = temp_magazine
    
    for i in range(start_switch_index):
        job_id = machine_jobs[i]
        elapsed_time += job_costs[job_id]
    
    for i in range(start_switch_index, len(machine_jobs)):
        job_id = machine_jobs[i]
        job_tools = tools_requirements_matrix[:, job_id]
        
        needed_tools = np.logical_or(magazine, job_tools).astype(np.int64)
        tools_count = np.sum(needed_tools)
        
        if tools_count > magazine_capacity:
            excess = tools_count - magazine_capacity
            num_switches = excess
            
            priority_tools = job_tools.copy()
            remaining_capacity = magazine_capacity - np.sum(job_tools)
            
            if remaining_capacity > 0:
                available_old_tools = magazine & ~job_tools
                old_tools_indices = np.where(available_old_tools)[0]
                
                for j, tool_idx in enumerate(old_tools_indices):
                    if j < remaining_capacity:
                        priority_tools[tool_idx] = 1
                    else:
                        break
            
            magazine = priority_tools
        else:
            num_switches = 0
            magazine = needed_tools
        
        elapsed_time += num_switches * tool_change_cost + job_costs[job_id]
    
    return elapsed_time


@njit
def calculate_makespan_all_machines(job_assignment, tools_requirements_matrix, magazines_capacities, 
                                    tool_change_costs, job_cost_per_machine):
    """Calcula o makespan de todas as máquinas."""
    makespans = np.zeros(len(magazines_capacities), dtype=np.int64)
    
    for machine_id in range(len(magazines_capacities)):
        machine_jobs = job_assignment[machine_id][job_assignment[machine_id] != -1]
        makespans[machine_id] = calculate_makespan_for_machine(
            machine_jobs, 
            tools_requirements_matrix, 
            magazines_capacities[machine_id],
            tool_change_costs[machine_id],
            job_cost_per_machine[machine_id]
        )
    
    return makespans


@njit
def get_system_makespan(makespans):
    """Retorna o makespan do sistema (máximo entre todas as máquinas)."""
    return np.max(makespans)
