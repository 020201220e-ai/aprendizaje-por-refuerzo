import streamlit as st
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

# Definición de la Red (debe ser igual a la del entrenamiento)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

st.title("DQN Agent - CartPole Demo")
st.write("Este agente fue entrenado usando Deep Q-Learning.")

# Cargar modelo
env = gym.make('CartPole-v1', render_mode='rgb_array')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = QNetwork(state_size, action_size)
try:
    model.load_state_dict(torch.load('dqn_cartpole_model.pth', map_location=torch.device('cpu')))
    st.success("Modelo cargado correctamente!")
except:
    st.error("No se encontró el archivo dqn_cartpole_model.pth")

if st.button('Ejecutar Episodio'):
    state, _ = env.reset()
    img = st.image([])
    total_reward = 0
    done = False
    
    while not done:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(model(state_t)).item()
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Mostrar frame
        img.image(env.render())
        time.sleep(0.02)
        
    st.write(f"Recompensa final: {total_reward}")