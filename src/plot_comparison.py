"""Gera gráficos de comparação GA vs VNS."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

comparison_file = r'c:\Users\igort\Programacao\CE\src\results\comparison_ga_vs_vns.csv'
df = pd.read_csv(comparison_file)

df_valid = df[df['ga_best_makespan'].notna() & df['vns_best_makespan'].notna()].copy()

if len(df_valid) > 0:
    # Gráfico 1: Comparação de makespan
    fig, ax = plt.subplots(figsize=(10, 6))
    
    instances = df_valid['instance'].str.replace('_m=.*', '', regex=True)
    x = np.arange(len(instances))
    width = 0.35
    
    ax.bar(x - width/2, df_valid['ga_best_makespan'], width, 
           label='GA', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, df_valid['vns_best_makespan'], width,
           label='VNS', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Instância', fontweight='bold')
    ax.set_ylabel('Makespan', fontweight='bold')
    ax.set_title('Comparação de Makespan: GA vs VNS', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\igort\Programacao\CE\src\results\ga_vs_vns_makespan.png', 
                dpi=300, bbox_inches='tight')
    print("Gráfico de makespan salvo em: src/results/ga_vs_vns_makespan.png")
    plt.close()
    
    # Gráfico 2: Gap percentual
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(instances, df_valid['ga_vns_gap_pct'], 
                  color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Instância', fontweight='bold')
    ax.set_ylabel('Gap (%)', fontweight='bold')
    ax.set_title('Gap Percentual do GA em relação ao VNS', fontsize=14, fontweight='bold')
    ax.set_xticklabels(instances, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(r'c:\Users\igort\Programacao\CE\src\results\ga_vs_vns_gap.png',
                dpi=300, bbox_inches='tight')
    print("Gráfico de gap salvo em: src/results/ga_vs_vns_gap.png")
    plt.close()

# Gráfico 3: Todos os resultados (incluindo os que não têm comparação)
fig, ax = plt.subplots(figsize=(12, 6))

instances_all = df['instance'].str.replace('_m=.*', '', regex=True)
x = np.arange(len(instances_all))
width = 0.35

ga_values = df['ga_best_makespan'].fillna(0)
vns_values = df['vns_best_makespan'].fillna(0)

bars_ga = ax.bar(x - width/2, ga_values, width, label='GA', color='#2E86AB', alpha=0.8)
bars_vns = ax.bar(x + width/2, vns_values, width, label='VNS', color='#A23B72', alpha=0.8)

ax.set_xlabel('Instância', fontweight='bold')
ax.set_ylabel('Makespan', fontweight='bold')
ax.set_title('Resultados GA e VNS (todas as instâncias testadas)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(instances_all, rotation=45, ha='right')
ax.legend(loc='upper left', frameon=True, shadow=True)
ax.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras e anotações para valores ausentes
for i, (ga, vns, bar_ga, bar_vns) in enumerate(zip(ga_values, vns_values, bars_ga, bars_vns)):
    if ga > 0:
        height = bar_ga.get_height()
        ax.text(bar_ga.get_x() + bar_ga.get_width()/2., height,
                f'{int(ga)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(i - width/2, 50, 'N/A', ha='center', va='bottom', 
                fontsize=8, fontweight='bold', color='red')
    
    if vns > 0:
        height = bar_vns.get_height()
        ax.text(bar_vns.get_x() + bar_vns.get_width()/2., height,
                f'{int(vns)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(i + width/2, 50, 'N/A', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(r'c:\Users\igort\Programacao\CE\src\results\ga_vs_vns_all.png',
            dpi=300, bbox_inches='tight')
print("Gráfico completo salvo em: src/results/ga_vs_vns_all.png")
plt.close()

# Gráfico 4: Tempo de execução e trade-off
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Tempo de execução (assumindo VNS = 68s * 30 runs = 2040s)
instances_time = ['ins1']
ga_time = [2.7]  # segundos
vns_time = [2040]  # 68s * 30 runs

ax1.bar([0], ga_time, width=0.4, label='GA (1 run)', color='#2E86AB', alpha=0.8)
ax1.bar([1], vns_time, width=0.4, label='VNS (30 runs)', color='#A23B72', alpha=0.8)
ax1.set_ylabel('Tempo Total (segundos)', fontweight='bold')
ax1.set_title('Tempo Computacional - ins1', fontsize=12, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['GA', 'VNS'])
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
ax1.text(0, ga_time[0], f'{ga_time[0]:.1f}s', ha='center', va='bottom', fontweight='bold')
ax1.text(1, vns_time[0], f'{vns_time[0]:.0f}s\n(34 min)', ha='center', va='bottom', fontweight='bold')

# Trade-off qualidade vs tempo
methods = ['GA', 'VNS']
quality = [36, 32]  # makespan
time_normalized = [1, 756]  # normalizado (GA = 1)

ax2.scatter(time_normalized, quality, s=300, alpha=0.6, 
           c=['#2E86AB', '#A23B72'], edgecolors='black', linewidth=2)

for i, method in enumerate(methods):
    ax2.annotate(method, (time_normalized[i], quality[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax2.set_xlabel('Tempo Relativo (GA = 1)', fontweight='bold')
ax2.set_ylabel('Makespan', fontweight='bold')
ax2.set_title('Trade-off Qualidade vs Tempo - ins1', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()  # Menor makespan é melhor

plt.tight_layout()
plt.savefig(r'c:\Users\igort\Programacao\CE\src\results\ga_vs_vns_tradeoff.png',
            dpi=300, bbox_inches='tight')
print("Gráfico de trade-off salvo em: src/results/ga_vs_vns_tradeoff.png")
plt.close()

# Gráfico 5: Comparação detalhada apenas ins1 (para melhor visualização)
if len(df_valid) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filtrar apenas ins1
    df_ins1 = df_valid[df_valid['instance'].str.contains('ins1_m=2')]
    
    if len(df_ins1) > 0:
        categories = ['GA', 'VNS']
        values = [df_ins1['ga_best_makespan'].values[0], df_ins1['vns_best_makespan'].values[0]]
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}',
                    ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Adicionar linha de diferença
        ax.plot([0, 1], [values[0], values[1]], 'k--', linewidth=1.5, alpha=0.5)
        ax.text(0.5, (values[0] + values[1])/2, f'Gap: {int(values[0] - values[1])} (12.5%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
        
        ax.set_ylabel('Makespan', fontweight='bold', fontsize=12)
        ax.set_title('Comparação Detalhada - Instância ins1\n(2 máquinas, 10 tarefas)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylim([0, max(values) * 1.2])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(r'c:\Users\igort\Programacao\CE\src\results\ga_vs_vns_ins1_detail.png',
                    dpi=300, bbox_inches='tight')
        print("Gráfico detalhado ins1 salvo em: src/results/ga_vs_vns_ins1_detail.png")
        plt.close()

print("\n✅ Todos os gráficos foram gerados com sucesso!")
