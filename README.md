# Projeto CartPole - Controle Manual e Aprendizado por Reforço

Este repositório contém a implementação do **Projeto 1**, que inclui:

1. **Controle Manual (Bang-Bang):** Um operador humano tenta equilibrar o pêndulo usando as teclas do teclado.
2. **Aprendizado por Reforço:** Comparação entre dois algoritmos clássicos — **Q-Learning Tabular** e **DQN (Deep Q-Network)**.

Ambos utilizam o ambiente **CartPole-v1** da biblioteca Gymnasium.

---

## 📋 Pré-requisitos

Para executar este código, você precisará do **Python 3.7+** instalado.

As dependências necessárias são:
- `gymnasium`: Para o ambiente de simulação e física.
- `pygame`: Para capturar eventos do teclado e gerenciar a janela de renderização.
- `numpy`: Para cálculos matemáticos auxiliares.
- `matplotlib`: Para geração de gráficos comparativos.
- `torch` (PyTorch): Para implementação da rede neural do DQN.

---

## 🚀 Instalação

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/rossatodias/GymnasiumCartPole.git
cd GymnasiumCartPole
```

### 2️⃣ Criar e ativar um ambiente virtual (recomendado)
```bash
python3 -m venv venv        # Linux/macOS
source venv/bin/activate    # Linux/macOS

python -m venv venv         # Windows
venv\Scripts\activate       # Windows
```

### 3️⃣ Instalar as dependências dentro do venv
```bash
pip install -r requirements.txt
```

---

## 🎮 Modo 1: Controle Manual (`human.py`)

Este modo permite que um operador humano tente equilibrar o pêndulo utilizando as teclas do teclado. Para tornar a tarefa viável e didática, considerando o tempo de reação humano, foram aplicadas modificações na física e na velocidade de renderização.

### Como Executar

```bash
python3 human.py        # Linux/macOS
python human.py         # Windows
```

### 🕹️ Controles

A mecânica de controle é do tipo **Bang-Bang**, onde uma força fixa é aplicada em uma direção.

- **Seta Esquerda (←):** Aplica força máxima para a esquerda.
- **Seta Direita (→):** Aplica força máxima para a direita.
- **Tecla Q:** Encerra a simulação e exibe o resumo das recompensas.

> **Nota:** Se nenhuma tecla for pressionada, o ambiente continuará aplicando a última ação ou o padrão, por isso são necessárias correções constantes.

### ⚙️ Configurações

Para facilitar o controle por humanos, foram aplicados os seguintes ajustes:

| Parâmetro | Valor | Motivo |
|-----------|-------|--------|
| **FPS** | 10 (Slow Motion) | Compensa o delay de reação visual-motor humano |
| **Limite de Ângulo** | 45° (padrão: 12°) | Oferece janela de recuperação maior |

---

## 🤖 Modo 2: Aprendizado por Reforço (`rlearning.py`)

Este modo treina e compara dois algoritmos de aprendizado por reforço:

### Algoritmos Implementados

#### Q-Learning Tabular
- Discretiza o espaço de estados contínuo em bins
- Armazena Q-values em uma tabela multidimensional
- Ideal para espaços de estados pequenos/médios
- **Tamanho da tabela:** ~16.562 valores

#### DQN (Deep Q-Network)
- Utiliza rede neural para aproximar a função Q
- **Experience Replay:** Armazena transições para treinamento em batch
- **Target Network:** Rede separada para estabilidade do treinamento
- Ideal para espaços de estados contínuos/grandes

### Como Executar

```bash
python3 rlearning.py        # Linux/macOS
python rlearning.py         # Windows
```

O script irá:
1. Treinar o agente Q-Learning por 650 episódios
2. Treinar o agente DQN por 650 episódios
3. Gerar um gráfico comparativo (`comparacao_rl.png`)
4. Demonstrar visualmente ambos os agentes treinados

### ⚙️ Hiperparâmetros

Os principais hiperparâmetros podem ser ajustados no início do arquivo:

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| `EPISODES` | 650 | Total de episódios de treinamento |
| `MAX_STEPS` | 500 | Limite de passos por episódio |
| `GAMMA` | 0.99 | Fator de desconto |
| `EPSILON_START` | 1.0 | Exploração inicial (100% aleatório) |
| `EPSILON_END` | 0.01 | Exploração final (1% aleatório) |
| `EPSILON_DECAY` | 0.995 | Taxa de decaimento da exploração |
| `LEARNING_RATE_Q` | 0.1 | Taxa de aprendizado (Q-Learning) |
| `LEARNING_RATE_DQN` | 0.001 | Taxa de aprendizado (DQN) |
| `BATCH_SIZE` | 64 | Tamanho do batch (DQN) |

### 📈 Saída

- **Terminal:** Progresso do treinamento a cada 50 episódios (média, máximo, epsilon)
- **Gráfico:** Arquivo `comparacao_rl.png` com curvas de aprendizado suavizadas
- **Demonstração:** Execução visual de ambos os agentes após o treinamento

---

## 📊 Estrutura do Projeto

```
GymnasiumCartPole/
├── human.py           # Controle manual (bang-bang)
├── rlearning.py       # Aprendizado por reforço (Q-Learning + DQN)
├── requirements.txt   # Dependências do projeto
├── comparacao_rl.png  # Gráfico gerado (após execução)
└── README.md          # Este arquivo
```

---

## 🔧 Solução de Problemas

### VS Code
O VS Code deve perguntar: *"We noticed a new environment... do you want to select it?"* → Clique em **Yes**. Se não perguntar, faça o passo abaixo e selecione o `venv` que você acabou de criar.

### Selecione o Interpretador Correto no VS Code
Se você não selecionou a opção anterior, provavelmente o VS Code estará olhando para um Python diferente do que você usou para instalar.

1. Abra seu arquivo `.py` no VS Code.
2. Pressione **`Ctrl + Shift + P`** (ou `Cmd + Shift + P` no Mac).
3. Digite e selecione: **`Python: Select Interpreter`**.
4. Vai aparecer uma lista.
   - Se você criou um ambiente virtual (pasta `.venv` ou `venv`), selecione a opção que tem `./venv/bin/python` ou similar.
   - Se não criou, procure a versão "Global" onde você rodou o `pip install`.
   - *Dica: Geralmente a opção "Recommended" é a correta.*

---
