import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# CONFIGURAÇÕES GERAIS
EPISODES = 650              # Total de episódios de treinamento
MAX_STEPS = 500             # Limite máximo de passos (padrão CartPole-v1)
GAMMA = 0.99                # Fator de desconto (importância de recompensas futuras)
EPSILON_START = 1.0         # Exploração inicial (100% aleatório)
EPSILON_END = 0.01          # Exploração final (1% aleatório)
EPSILON_DECAY = 0.995       # Taxa de decaimento da exploração
LEARNING_RATE_Q = 0.1       # Taxa de aprendizado para Q-Learning tabular
LEARNING_RATE_DQN = 0.001   # Taxa de aprendizado para DQN (rede neural)
BATCH_SIZE = 64             # Tamanho do batch para treinamento DQN
UPDATE_EVERY = 4            # Frequência de atualização da policy network
TARGET_UPDATE = 500         # Frequência de atualização da target network
MIN_REPLAY_SIZE = 1000      # Experiências mínimas antes de iniciar treinamento


# CLASSE: Q-LEARNING TABULAR (DISCRETIZAÇÃO DO ESPAÇO CONTÍNU
class QLearningAgent:
    def __init__(self, env):
        self.env = env

        # Define os bins para discretização de cada dimensão do estado
        # Mais bins = maior granularidade, mas tabela maior
        self.bins = [
            np.linspace(-2.4, 2.4, 6),      # Posição: [-2.4, 2.4] → 6 bins
            np.linspace(-3.0, 3.0, 6),      # Velocidade: [-3, 3] → 6 bins
            np.linspace(-0.209, 0.209, 12), # Ângulo: [-12°, 12°] → 12 bins (CRÍTICO)
            np.linspace(-3.0, 3.0, 12)      # Vel. Angular: [-3, 3] → 12 bins (CRÍTICO)
        ]
        
        # Cria a Q-Table: dimensões = (bins+1 para cada estado) + ações
        # Tamanho: 7 x 7 x 13 x 13 x 2 = 16,562 valores
        self.q_table = np.zeros(
            tuple(len(b) + 1 for b in self.bins) + (env.action_space.n,)
        )
        
        # Parâmetro de exploração (epsilon-greedy)
        self.epsilon = EPSILON_START

    def discretize(self, state):
        indices = []
        for i, val in enumerate(state):
            # np.digitize retorna o índice do bin onde o valor se encaixa
            indices.append(np.digitize(val, self.bins[i]))
        return tuple(indices)

    def select_action(self, state_idx, training=True):
        # Durante treinamento: com probabilidade epsilon, explora aleatoriamente
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Caso contrário, escolhe a ação com maior Q-value (exploitação)
        return np.argmax(self.q_table[state_idx])

    def update(self, state_idx, action, reward, next_state_idx, done):
        # Calcula o valor alvo (target)
        # Se done=True, não há próximo estado → target = reward
        # Caso contrário: target = reward + desconto * melhor Q futuro
        target = reward + (0 if done else GAMMA * np.max(self.q_table[next_state_idx]))
        
        # Calcula o erro TD (Temporal Difference)
        current_q = self.q_table[state_idx + (action,)]
        error = target - current_q
        
        # Atualiza Q-value com taxa de aprendizado
        self.q_table[state_idx + (action,)] += LEARNING_RATE_Q * error

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# CLASSE: REDE NEURAL PARA DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),   # Camada de entrada
            nn.ReLU(),                    # Ativação não-linear
            nn.Linear(128, 128),          # Camada escondida
            nn.ReLU(),
            nn.Linear(128, output_dim)    # Camada de saída (Q-values)
        )

    def forward(self, x):
        return self.net(x)


# CLASSE: AGENTE DQN COM EXPERIENCE REPLAY E TARGET NETWORK
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]   # 4 para CartPole
        self.action_dim = env.action_space.n              # 2 para CartPole
        
        # Duas redes: policy (treina) e target (estável)
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        
        # Copia pesos iniciais para a target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Otimizador Adam para backpropagation
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=LEARNING_RATE_DQN
        )
        
        # Replay Buffer: armazena experiências (s, a, r, s', done)
        self.memory = deque(maxlen=10000)
        
        # Controle de exploração e contadores
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.update_counter = 0

    def select_action(self, state, training=True):
        # Exploração: ação aleatória
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Exploitação: ação com maior Q-value previsto pela rede
        state_t = torch.FloatTensor(state).unsqueeze(0)  # [1, 4]
        with torch.no_grad():  # Não calcula gradientes (inferência)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # Condições para atualizar
        if len(self.memory) < MIN_REPLAY_SIZE:
            return  # Aguarda experiências suficientes
        
        if self.steps_done % UPDATE_EVERY != 0:
            return  # Atualiza apenas a cada UPDATE_EVERY passos
        
        # Amostra mini-batch aleatório
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converte para tensores PyTorch
        states_t = torch.FloatTensor(np.array(states))          # [64, 4]
        actions_t = torch.LongTensor(actions).unsqueeze(1)      # [64, 1]
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)     # [64, 1]
        next_states_t = torch.FloatTensor(np.array(next_states))# [64, 4]
        dones_t = torch.FloatTensor(dones).unsqueeze(1)         # [64, 1]

        # Q-values atuais: Q(s, a) usando policy network
        q_values = self.policy_net(states_t).gather(1, actions_t)
        
        # Q-values alvo: r + γ·max Q(s', a') usando target network
        with torch.no_grad():  # Target network não treina aqui
            next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            # Se done=1, não há Q futuro → multiplica por (1 - done)
            target_q_values = rewards_t + (GAMMA * next_q_values * (1 - dones_t))

        # Calcula perda MSE e faz backpropagation
        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()              # Zera gradientes antigos
        loss.backward()                         # Calcula gradientes
        torch.nn.utils.clip_grad_norm_(         # Limita gradientes
            self.policy_net.parameters(), 1.0
        )
        self.optimizer.step()                   # Atualiza pesos

        self.update_counter += 1
        
        # Atualiza target network periodicamente para estabilidade
        if self.update_counter % (TARGET_UPDATE // UPDATE_EVERY) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  → Target network atualizada (update #{self.update_counter})")

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# FUNÇÃO: LOOP DE TREINAMENTO GENÉRICO
def train_agent(agent_type, episodes):
    # Cria ambiente
    env = gym.make("CartPole-v1")
    
    # Instancia agente baseado no tipo
    if agent_type == 'Q-Table':
        agent = QLearningAgent(env)
    else:
        agent = DQNAgent(env)

    rewards_history = []
    
    # Cabeçalho informativo
    print(f"\n{'='*60}")
    print(f"Iniciando treinamento: {agent_type}")
    print(f"{'='*60}")
    
    # Exibe configurações específicas
    if agent_type == 'DQN':
        print(f"Configurações DQN:")
        print(f"  - Atualização a cada {UPDATE_EVERY} passos")
        print(f"  - Target network a cada {TARGET_UPDATE} passos")
        print(f"  - Replay mínimo: {MIN_REPLAY_SIZE} experiências")
        print(f"{'='*60}\n")
    else:
        print(f"Configurações Q-Learning:")
        print(f"  - Bins: {[len(b) + 1 for b in agent.bins]}")
        print(f"  - Tamanho da tabela: {agent.q_table.size} valores")
        print(f"{'='*60}\n")
    
    # Loop principal de treinamento
    for ep in range(episodes):
        # Reseta ambiente
        state, _ = env.reset()
        if agent_type == 'Q-Table': 
            state = agent.discretize(state)  # Discretiza para Q-Learning
        
        total_reward = 0
        done = False
        
        # Loop do episódio
        while not done:
            # Seleciona e executa ação
            action = agent.select_action(state)
            next_state_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Atualiza agente
            if agent_type == 'Q-Table':
                # Q-Learning: atualiza diretamente a tabela
                next_state = agent.discretize(next_state_raw)
                agent.update(state, action, reward, next_state, done)
            else:
                # DQN: armazena experiência e treina em batch
                next_state = next_state_raw
                agent.store_transition(state, action, reward, next_state, done)
                agent.steps_done += 1
                agent.update()
            
            # Avança para próximo estado
            state = next_state
            total_reward += reward

        # Decai epsilon ao final do episódio
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # Logging a cada 50 episódios
        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            max_reward = np.max(rewards_history[-50:])
            print(f"Ep {ep+1:3d}/{episodes} | "
                  f"Avg(50): {avg:6.1f} | "
                  f"Max(50): {max_reward:3.0f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    
    # Resumo final
    print(f"\n{'='*60}")
    print(f"Treinamento {agent_type} concluído!")
    print(f"Média final (últimos 50): {np.mean(rewards_history[-50:]):.1f}")
    print(f"{'='*60}\n")
    
    return agent, rewards_history


# EXECUÇÃO PRINCIPAL: TREINAMENTO, COMPARAÇÃO E DEMONSTRAÇÃO
if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPARAÇÃO: Q-LEARNING TABULAR vs DQN")
    print("="*60)
    
    # 1. Treinar Q-Learning Tabular
    q_agent, q_rewards = train_agent('Q-Table', EPISODES)
    
    # 2. Treinar DQN
    dqn_agent, dqn_rewards = train_agent('DQN', EPISODES)

    # 3. GRÁFICO COMPARATIVO
    plt.figure(figsize=(12, 6))
    
    def moving_average(data, window_size=20):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Suaviza curvas para melhor visualização
    q_smooth = moving_average(q_rewards, 20)
    dqn_smooth = moving_average(dqn_rewards, 20)
    
    # Plota curvas de aprendizado
    plt.plot(q_smooth, color='#2E86AB', label='Q-Learning (Tabular)', 
             alpha=0.8, linewidth=2)
    plt.plot(dqn_smooth, color='#A23B72', label='DQN (Deep Learning)', 
             alpha=0.8, linewidth=2)
    
    # Linha de referência: objetivo do CartPole-v1
    plt.axhline(y=475, color='#F18F01', linestyle='--', 
                label='Objetivo CartPole-v1 (475)', alpha=0.7)
    
    # Configurações do gráfico
    plt.title('Comparativo de Desempenho: Q-Learning vs DQN', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Episódios', fontsize=12)
    plt.ylabel('Recompensa (Média Móvel 20)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salva gráfico
    plt.savefig('comparacao_rl.png', dpi=150)
    print("\n✓ Gráfico salvo: comparacao_rl.png")

    # 4. DEMONSTRAÇÃO VISUAL DOS AGENTES TREINADOS
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO VISUAL")
    print("="*60)
    print("Executando agentes treinados...\n")
    
    demos = [('Q-Table', q_agent), ('DQN', dqn_agent)]
    
    for name, agent in demos:
        print(f"\n→ Demonstrando {name}")
        
        # Cria ambiente com renderização visual
        env_visual = gym.make("CartPole-v1", render_mode="human")
        state, _ = env_visual.reset()
        
        # Discretiza se for Q-Learning
        if name == 'Q-Table': 
            state = agent.discretize(state)
        
        done = False
        total = 0
        steps = 0
        
        # Executa episódio visual (sem treinamento)
        while not done and steps < MAX_STEPS:
            action = agent.select_action(state, training=False)  # Greedy puro
            
            next_state_raw, reward, terminated, truncated, _ = env_visual.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
            
            # Atualiza estado
            if name == 'Q-Table':
                state = agent.discretize(next_state_raw)
            else:
                state = next_state_raw
        
        print(f"  Score: {total} | Steps: {steps}")
        env_visual.close()
    
    # Mensagem final
    print("\n" + "="*60)
    print("EXECUÇÃO COMPLETA!")
    print("="*60)