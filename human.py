import gymnasium as gym
import pygame
import numpy as np
import math

def manual_control():
    # --- CONFIGURAÇÃO DE DIFICULDADE ---
    FPS = 10                # Frames por segundo (Menor = Mais lento/Fácil)
    TARGET_ANGLE_DEG = 45   # Limite do ângulo em graus (Padrão é 12)
    # -----------------------------------

    # Cria o ambiente com renderização
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Modificação dos limites físicos
    new_limit_radians = TARGET_ANGLE_DEG * 2 * math.pi / 360
    env.unwrapped.theta_threshold_radians = new_limit_radians
    
    # Inicializa o Clock do Pygame para controlar o FPS
    clock = pygame.time.Clock()

    print(f"\n--- INICIANDO SIMULAÇÃO ---")
    print(f"Velocidade: {FPS} passos/segundo (Slow Motion)")
    print(f"Limite de Ângulo: {TARGET_ANGLE_DEG}°")
    print("Controles: SETA ESQUERDA / SETA DIREITA")
    print("Sair: Q")
    
    observation, info = env.reset()
    
    running = True
    action = 0 
    total_reward = 0
    episode_count = 0
    rewards = []

    while running:
        # Limita a velocidade do loop para o FPS definido
        # Isso dá tempo para o cérebro humano processar e reagir
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_q:
                    running = False

        # Passo do ambiente
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Verifica fim do episódio
        if terminated or truncated:
            cart_pos = observation[0]
            pole_angle = observation[2]
            
            fail_reason = "Sucesso (Tempo Esgotado)"
            if terminated:
                if abs(cart_pos) > env.unwrapped.x_threshold:
                    fail_reason = "Saiu da Pista"
                elif abs(pole_angle) > env.unwrapped.theta_threshold_radians:
                    fail_reason = "Caiu (Ângulo)"
            
            print(f"Episódio {episode_count + 1}: {total_reward} pontos. Causa: {fail_reason}")
            
            rewards.append(total_reward)
            episode_count += 1
            total_reward = 0
            observation, info = env.reset()

    env.close()
    
    if rewards:
        print(f"\nResumo:")
        print(f"Média: {np.mean(rewards):.2f}")
        print(f"Máximo: {np.max(rewards)}")

if __name__ == "__main__":
    manual_control()