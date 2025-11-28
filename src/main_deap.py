"""Algoritmo Genético para o problema SSP-NPM usando DEAP.

Este módulo implementa um AG monoobjetivo para minimização de makespan,
processando instâncias SSP-NPM-I e SSP-NPM-II em lote.
"""

import os
import sys
import time
import random
import csv
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '..'))

from functions.input import read_problem_instance
from functions.metaheuristics import (
    calculate_makespan_all_machines,
    get_system_makespan,
    check_machine_eligibility
)

INSTANCES_BASE_DIR = os.path.join(BASE_DIR, "../instances")
RESULTS_BASE_DIR = os.path.join(BASE_DIR, "results")
POPULATION_SIZE = 100
GENERATIONS = 1000
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
TOURNAMENT_SIZE = 3

INSTANCE_FOLDERS = ["SSP-NPM-I", "SSP-NPM-II"]

os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

instance_data = None

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def create_individual():
    """Cria um indivíduo usando heurística gulosa."""
    num_machines = instance_data["num_machines"]
    num_jobs = instance_data["num_jobs"]
    magazines_capacities = instance_data["magazines_capacities"]
    tools_per_job = instance_data["tools_per_job"]
    
    jobs_list = np.random.permutation(num_jobs)
    job_assignment = np.full((num_machines, num_jobs), -1, dtype=np.int64)
    
    for job_id in jobs_list:
        machine_loads = []
        for m in range(num_machines):
            if check_machine_eligibility(job_id, m, magazines_capacities, tools_per_job):
                load = np.sum(job_assignment[m] != -1)
                machine_loads.append((load, m))
        
        if machine_loads:
            _, selected_machine = min(machine_loads)
            pos = np.where(job_assignment[selected_machine] == -1)[0][0]
            job_assignment[selected_machine, pos] = job_id
    
    return creator.Individual(job_assignment.tolist())


def evaluate_makespan(individual):
    """Avalia o makespan de uma solução."""
    job_assignment = np.array(individual, dtype=np.int64)
    
    makespans = calculate_makespan_all_machines(
        job_assignment,
        instance_data["tools_requirements_matrix"],
        instance_data["magazines_capacities"],
        instance_data["tool_change_costs"],
        instance_data["job_cost_per_machine"]
    )
    
    return (int(get_system_makespan(makespans)),)


def mutate_swap(individual):
    """Mutação: troca dois jobs dentro da mesma máquina."""
    individual_array = np.array(individual, dtype=np.int64)
    num_machines = individual_array.shape[0]
    
    machine_id = random.randint(0, num_machines - 1)
    valid_jobs = np.where(individual_array[machine_id] != -1)[0]
    
    if len(valid_jobs) >= 2:
        pos1, pos2 = random.sample(list(valid_jobs), 2)
        individual_array[machine_id, pos1], individual_array[machine_id, pos2] = \
            individual_array[machine_id, pos2], individual_array[machine_id, pos1]
    
    for i in range(len(individual)):
        individual[i] = individual_array[i].tolist()
    
    return (individual,)


def crossover(ind1, ind2):
    """Crossover que troca uma máquina entre indivíduos e repara duplicatas."""
    arr1 = np.array(ind1, dtype=np.int64)
    arr2 = np.array(ind2, dtype=np.int64)

    num_machines, num_positions = arr1.shape
    num_jobs = instance_data["num_jobs"]

    machine_to_swap = random.randint(0, num_machines - 1)

    child1 = arr1.copy()
    child2 = arr2.copy()

    child1[machine_to_swap], child2[machine_to_swap] = (
        arr2[machine_to_swap].copy(),
        arr1[machine_to_swap].copy(),
    )

    def remove_duplicates(matrix):
        seen = set()
        for i in range(num_machines):
            for j in range(num_positions):
                job = matrix[i][j]
                if job == -1:
                    continue
                if job in seen:
                    matrix[i][j] = -1
                else:
                    seen.add(job)
        return seen

    seen1 = remove_duplicates(child1)
    seen2 = remove_duplicates(child2)

    all_jobs = set(range(num_jobs))
    missing1 = list(all_jobs - seen1)
    missing2 = list(all_jobs - seen2)

    def reinsert_missing(matrix, missing_jobs):
        for job in missing_jobs:
            candidates = []
            for m in range(num_machines):
                if check_machine_eligibility(
                    job, m,
                    instance_data["magazines_capacities"],
                    instance_data["tools_per_job"],
                ):
                    load = np.sum(matrix[m] != -1)
                    candidates.append((load, m))

            if not candidates:
                for m in range(num_machines):
                    free = np.where(matrix[m] == -1)[0]
                    if len(free) > 0:
                        matrix[m][free[0]] = job
                        break
                continue

            _, best_machine = min(candidates)
            free_positions = np.where(matrix[best_machine] == -1)[0]

            if len(free_positions) > 0:
                matrix[best_machine][free_positions[0]] = job
            else:
                for m in range(num_machines):
                    free = np.where(matrix[m] == -1)[0]
                    if len(free) > 0:
                        matrix[m][free[0]] = job
                        break

    reinsert_missing(child1, missing1)
    reinsert_missing(child2, missing2)

    return creator.Individual(child1.tolist()), creator.Individual(child2.tolist())


toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_makespan)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate_swap)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)


def plot_evolution(logbook, save_path, instance_name):
    """Plota a evolução do fitness ao longo das gerações."""
    try:
        import matplotlib.pyplot as plt
        
        gen = logbook.select("gen")
        min_fits = logbook.select("min")
        avg_fits = logbook.select("avg")
        max_fits = logbook.select("max")
        
        plt.figure(figsize=(12, 7))
        plt.plot(gen, min_fits, 'b-', label='Melhor', linewidth=2, marker='o', markersize=3)
        plt.plot(gen, avg_fits, 'g--', label='Média', linewidth=1.5)
        plt.plot(gen, max_fits, 'r:', label='Pior', linewidth=1, alpha=0.7)
        
        plt.xlabel('Geração', fontsize=12)
        plt.ylabel('Makespan', fontsize=12)
        plt.title(f'Evolução do Fitness - AG\n{instance_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  ✓ Gráfico salvo: {os.path.basename(save_path)}")
    except ImportError:
        print("  ⚠ Matplotlib não disponível")
    except Exception as e:
        print(f"  ⚠ Erro ao gerar gráfico: {e}")


def save_evolution_csv(logbook, save_path, instance_name, execution_time, best_makespan):
    """Salva dados de evolução em CSV."""
    try:
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Cabeçalho
            writer.writerow(['instance', 'generation', 'min', 'avg', 'max', 'std', 
                            'execution_time', 'best_makespan'])
            
            for record in logbook:
                writer.writerow([
                    instance_name, record['gen'], record['min'], record['avg'],
                    record['max'], record['std'],
                    execution_time if record['gen'] == len(logbook) - 1 else '',
                    best_makespan if record['gen'] == len(logbook) - 1 else ''
                ])
        print(f"  ✓ Dados de evolução salvos: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"  ⚠ Erro ao salvar CSV: {e}")


def save_best_solution(best_individual, save_path, instance_name, best_makespan, execution_time):
    """Salva a melhor solução encontrada."""
    try:
        best_array = np.array(best_individual, dtype=np.int64)
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['instance', 'makespan', 'execution_time', 'machine_id', 'jobs'])
            
            for machine_id in range(best_array.shape[0]):
                jobs = best_array[machine_id][best_array[machine_id] != -1]
                jobs_str = ';'.join(map(str, jobs.tolist()))
                writer.writerow([instance_name, best_makespan, execution_time, machine_id, jobs_str])
        
        print(f"  ✓ Melhor solução salva: {os.path.basename(save_path)}")
    except Exception as e:
        print(f"  ⚠ Erro ao salvar solução: {e}")


def run_ga_for_instance(instance_path, instance_name, output_dir):
    """Executa o algoritmo genético para uma instância específica"""
    global instance_data
    
    print(f"\n{'='*70}")
    print(f"  Processando: {instance_name}")
    print(f"{'='*70}")
    
    # Carrega a instância
    try:
        instance_data = read_problem_instance(instance_path)
        print(f"  ✓ Instância carregada")
        print(f"    - Máquinas: {instance_data['num_machines']}")
        print(f"    - Jobs: {instance_data['num_jobs']}")
        print(f"    - Ferramentas: {instance_data['num_tools']}")
    except Exception as e:
        print(f"  ✗ Erro ao carregar instância: {e}")
        return None
    
    # Inicializa população
    random.seed(42)
    np.random.seed(42)
    
    population = toolbox.population(n=POPULATION_SIZE)
    
    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("std", np.std)
    
    # Hall of Fame (melhores indivíduos)
    hof = tools.HallOfFame(1)
    
    # Executa o algoritmo
    print(f"\n  Executando AG (Pop={POPULATION_SIZE}, Gen={GENERATIONS})...")
    start_time = time.time()
    
    try:
        population, logbook = algorithms.eaSimple(
            population, 
            toolbox,
            cxpb=CROSSOVER_PROB,
            mutpb=MUTATION_PROB,
            ngen=GENERATIONS,
            stats=stats,
            halloffame=hof,
            verbose=False  # Desativa output verboso
        )
        
        execution_time = time.time() - start_time
        
        # Resultados
        best_makespan = hof[0].fitness.values[0]
        initial_makespan = logbook[0]['min']
        improvement = initial_makespan - best_makespan
        improvement_pct = (improvement / initial_makespan * 100) if initial_makespan > 0 else 0
        
        print(f"\n  Resultados:")
        print(f"    ⏱  Tempo de execução: {execution_time:.2f}s")
        print(f"    🎯 Makespan inicial: {initial_makespan}")
        print(f"    ✨ Melhor makespan: {best_makespan}")
        print(f"    📈 Melhoria: {improvement} ({improvement_pct:.2f}%)")
        
        # Salva resultados
        print(f"\n  Salvando resultados em: {output_dir}")
        
        # 1. Gráfico de evolução
        plot_path = os.path.join(output_dir, "evolution.png")
        plot_evolution(logbook, save_path=plot_path, instance_name=instance_name)
        
        # 2. CSV com dados de evolução
        csv_path = os.path.join(output_dir, "evolution_data.csv")
        save_evolution_csv(logbook, csv_path, instance_name, execution_time, best_makespan)
        
        # 3. Melhor solução
        solution_path = os.path.join(output_dir, "best_solution.csv")
        save_best_solution(hof[0], solution_path, instance_name, best_makespan, execution_time)
        
        # 4. Resumo
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Instância: {instance_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Parâmetros do AG:\n")
            f.write(f"  - População: {POPULATION_SIZE}\n")
            f.write(f"  - Gerações: {GENERATIONS}\n")
            f.write(f"  - Prob. Crossover: {CROSSOVER_PROB}\n")
            f.write(f"  - Prob. Mutação: {MUTATION_PROB}\n\n")
            f.write(f"Dados da Instância:\n")
            f.write(f"  - Máquinas: {instance_data['num_machines']}\n")
            f.write(f"  - Jobs: {instance_data['num_jobs']}\n")
            f.write(f"  - Ferramentas: {instance_data['num_tools']}\n\n")
            f.write(f"Resultados:\n")
            f.write(f"  - Tempo de execução: {execution_time:.2f}s\n")
            f.write(f"  - Makespan inicial: {initial_makespan}\n")
            f.write(f"  - Melhor makespan: {best_makespan}\n")
            f.write(f"  - Melhoria: {improvement} ({improvement_pct:.2f}%)\n\n")
            f.write(f"Melhor solução encontrada:\n")
            best_array = np.array(hof[0], dtype=np.int64)
            for machine_id in range(best_array.shape[0]):
                jobs = best_array[machine_id][best_array[machine_id] != -1]
                f.write(f"  Máquina {machine_id}: {jobs.tolist()}\n")
        
        print(f"  ✓ Resumo salvo: summary.txt")
        
        return {
            'instance': instance_name,
            'execution_time': execution_time,
            'initial_makespan': initial_makespan,
            'best_makespan': best_makespan,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'num_machines': instance_data['num_machines'],
            'num_jobs': instance_data['num_jobs'],
            'num_tools': instance_data['num_tools']
        }
        
    except Exception as e:
        print(f"  ✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_instance_folder(folder_name):
    """Processa todas as instâncias de uma pasta específica"""
    instances_dir = os.path.join(INSTANCES_BASE_DIR, folder_name)
    results_dir = os.path.join(RESULTS_BASE_DIR, folder_name)
    
    # Verifica se a pasta de instâncias existe
    if not os.path.exists(instances_dir):
        print(f"\n⚠ Pasta {folder_name} não encontrada em {INSTANCES_BASE_DIR}")
        return []
    
    print(f"\n{'='*70}")
    print(f"  PROCESSANDO: {folder_name}")
    print(f"{'='*70}")
    print(f"  📁 Origem: {instances_dir}")
    print(f"  💾 Destino: {results_dir}")
    
    # Cria diretório de resultados para esta pasta
    os.makedirs(results_dir, exist_ok=True)
    
    # Lista todas as instâncias
    instance_files = sorted([f for f in os.listdir(instances_dir) if f.endswith('.csv')])
    
    if not instance_files:
        print(f"\n✗ Nenhuma instância encontrada em {instances_dir}")
        return []
    
    print(f"\n✓ {len(instance_files)} instância(s) encontrada(s)")
    
    # Processa cada instância
    all_results = []
    folder_start_time = time.time()
    
    for idx, instance_file in enumerate(instance_files, 1):
        instance_path = os.path.join(instances_dir, instance_file)
        instance_name = os.path.splitext(instance_file)[0]
        
        # Cria diretório para a instância
        instance_output_dir = os.path.join(results_dir, instance_name)
        os.makedirs(instance_output_dir, exist_ok=True)
        
        # Executa o GA
        result = run_ga_for_instance(instance_path, instance_name, instance_output_dir)
        
        if result:
            result['folder'] = folder_name  # Adiciona identificador da pasta
            all_results.append(result)
        
        print(f"\n{'='*70}")
        print(f"  Progresso {folder_name}: {idx}/{len(instance_files)} instâncias processadas")
        print(f"{'='*70}")
    
    folder_time = time.time() - folder_start_time
    
    # Gera relatório consolidado para esta pasta
    if all_results:
        print(f"\n{'='*70}")
        print(f"  RELATÓRIO - {folder_name}")
        print(f"{'='*70}\n")
        
        consolidated_path = os.path.join(results_dir, f"consolidated_{folder_name}.csv")
        df = pd.DataFrame(all_results)
        df.to_csv(consolidated_path, index=False)
        print(f"✓ Resultados consolidados salvos: {consolidated_path}")
        
        print(f"\nEstatísticas de {folder_name}:")
        print(f"  - Total de instâncias processadas: {len(all_results)}")
        print(f"  - Tempo total de execução: {folder_time:.2f}s")
        print(f"  - Tempo médio por instância: {folder_time/len(all_results):.2f}s")
        print(f"  - Makespan médio inicial: {df['initial_makespan'].mean():.2f}")
        print(f"  - Makespan médio final: {df['best_makespan'].mean():.2f}")
        print(f"  - Melhoria média: {df['improvement_pct'].mean():.2f}%")
    
    return all_results


def main():
    """Executa o algoritmo genético para todas as pastas de instâncias"""
    print("\n" + "="*70)
    print("  ALGORITMO GENÉTICO MONOOBJETIVO - MINIMIZAÇÃO DE MAKESPAN")
    print("  Processamento em Lote de Instâncias SSP-NPM")
    print("="*70)
    print(f"\nConfiguração:")
    print(f"  - População: {POPULATION_SIZE}")
    print(f"  - Gerações: {GENERATIONS}")
    print(f"  - Crossover: {CROSSOVER_PROB}")
    print(f"  - Mutação: {MUTATION_PROB}")
    print(f"  - Pastas de instâncias: {', '.join(INSTANCE_FOLDERS)}")
    print(f"  - Diretório base de resultados: {RESULTS_BASE_DIR}")
    
    # Processa cada pasta de instâncias
    all_results_combined = []
    total_start_time = time.time()
    
    for folder_name in INSTANCE_FOLDERS:
        folder_results = process_instance_folder(folder_name)
        all_results_combined.extend(folder_results)
    
    total_time = time.time() - total_start_time
    
    # Gera relatório consolidado GERAL (todas as pastas)
    if all_results_combined:
        print(f"\n{'='*70}")
        print("  RELATÓRIO CONSOLIDADO GERAL")
        print(f"{'='*70}\n")
        
        consolidated_path = os.path.join(RESULTS_BASE_DIR, "consolidated_all_results.csv")
        df_all = pd.DataFrame(all_results_combined)
        df_all.to_csv(consolidated_path, index=False)
        print(f"✓ Resultados gerais consolidados salvos: {consolidated_path}")
        
        print(f"\nEstatísticas Gerais (Todas as Pastas):")
        print(f"  - Total de instâncias processadas: {len(all_results_combined)}")
        print(f"  - Tempo total de execução: {total_time:.2f}s")
        print(f"  - Tempo médio por instância: {total_time/len(all_results_combined):.2f}s")
        print(f"  - Makespan médio inicial: {df_all['initial_makespan'].mean():.2f}")
        print(f"  - Makespan médio final: {df_all['best_makespan'].mean():.2f}")
        print(f"  - Melhoria média: {df_all['improvement_pct'].mean():.2f}%")
        
        # Estatísticas por pasta
        print(f"\nEstatísticas por Pasta:")
        for folder in INSTANCE_FOLDERS:
            df_folder = df_all[df_all['folder'] == folder]
            if len(df_folder) > 0:
                print(f"\n  📁 {folder}:")
                print(f"     - Instâncias: {len(df_folder)}")
                print(f"     - Makespan médio inicial: {df_folder['initial_makespan'].mean():.2f}")
                print(f"     - Makespan médio final: {df_folder['best_makespan'].mean():.2f}")
                print(f"     - Melhoria média: {df_folder['improvement_pct'].mean():.2f}%")
        
        print(f"\n{'='*70}")
        print("  ✨ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
