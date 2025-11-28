# Algoritmo Genético para SSP-NPM - Versão Monoobjetivo

Este código implementa um **Algoritmo Genético (GA)** usando a biblioteca **DEAP** para resolver o problema de **Sequenciamento e Escalonamento com Máquinas Paralelas Não-Idênticas (SSP-NPM)**.

## 🎯 Objetivo

Minimizar o **makespan** (tempo máximo de conclusão) do sistema, considerando:
- Alocação de jobs às máquinas
- Sequenciamento de jobs em cada máquina
- Restrições de capacidade de magazine de ferramentas
- Custos de troca de ferramentas

## 🧬 Método: Algoritmo Genético com DEAP

### Por que DEAP?

**DEAP** (Distributed Evolutionary Algorithms in Python) é uma biblioteca robusta e eficiente para computação evolucionária que oferece:
- Implementações prontas de operadores genéticos
- Flexibilidade para customização
- Suporte a estatísticas e análise de convergência
- Fácil integração com problemas de otimização

### Estrutura do Algoritmo

1. **Representação do Indivíduo**
   - Matriz numpy de dimensão `(num_machines, num_jobs)`
   - Valores: IDs dos jobs ou -1 para posições vazias
   
2. **Função de Fitness**
   - Calcula o makespan da solução usando funções Numba otimizadas
   - Minimização: peso = -1.0

3. **Operadores Genéticos**
   - **Crossover PMX**: Partially Mapped Crossover adaptado
   - **Mutação Swap**: Troca aleatória de jobs dentro da mesma máquina
   - **Seleção por Torneio**: Seleciona os melhores indivíduos

4. **Parâmetros**
   - População: 50 indivíduos
   - Gerações: 100
   - Taxa de crossover: 80%
   - Taxa de mutação: 20%
   - Tamanho do torneio: 3

## 📦 Instalação

```bash
pip install -r requirements.txt
```

## 🚀 Execução

```bash
python main_deap.py
```

## 📊 Saída

O programa exibe:
- Estatísticas por geração (melhor, média, pior fitness)
- Tempo de execução total
- Melhor makespan encontrado
- Atribuição ótima de jobs às máquinas

## 🔧 Personalização

Você pode ajustar os parâmetros no início do arquivo:
- `POPULATION_SIZE`: Tamanho da população
- `GENERATIONS`: Número de gerações
- `CROSSOVER_PROB`: Probabilidade de crossover
- `MUTATION_PROB`: Probabilidade de mutação
- `TOURNAMENT_SIZE`: Tamanho do torneio de seleção
- `INSTANCE_PATH`: Caminho para a instância do problema

## 📚 Referências

- DEAP Documentation: https://deap.readthedocs.io/
- Problema SSP-NPM: Scheduling and Sequencing Problem with Non-identical Parallel Machines
