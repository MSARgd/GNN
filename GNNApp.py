#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 07:47:34 2024
@author: msa
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import networkx as nx
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# Define the Graph Neural Network (GNN) model using TensorFlow
class GNNModel(Model):
    def __init__(self, hidden_size, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = layers.Dense(hidden_size, activation='relu')
        self.conv2 = layers.Dense(num_classes)

    def call(self, x, edge_index):
        x = self.conv1(x)
        segment_ids = edge_index[:, 0]
        unique_segment_ids, segment_indices = tf.unique(segment_ids)
        segment_ids_mapped = tf.gather(segment_indices, segment_ids)
        x_aggregated = tf.math.unsorted_segment_sum(x, segment_ids_mapped, num_segments=tf.reduce_max(segment_ids_mapped) + 1)
        x = tf.nn.relu(x_aggregated)
        x = self.conv2(x)
        return x

# Define a function to preprocess the data and create a graph
def preprocess_data():
    num_agents = 6
    num_substations = 4

    agent_data = {
        'prod_energy': np.random.uniform(low=0.5, high=5, size=num_agents),
        'stock_energy': np.random.uniform(low=0, high=10, size=num_agents),
        'cons_energy': np.random.uniform(low=0.2, high=3, size=num_agents),
        'gestion_charge': np.random.choice([True, False], size=num_agents),
        'repartition_energy': np.random.uniform(low=0.1, high=0.9, size=num_agents),
        'reactivite_signaux': np.random.uniform(low=0.5, high=1, size=num_agents),
        'distance_transmission': np.random.uniform(low=1, high=10, size=num_agents),
        'prevision_demande': np.random.uniform(low=0.8, high=1, size=num_agents)
    }

    substation_data = {
        'connectivity': [np.random.randint(0, num_agents, size=np.random.randint(1, num_agents)).tolist() for _ in range(num_substations)]
    }

    G = nx.Graph()

    for i in range(num_agents):
        G.add_node(f'Agent_{i}', **{k: v[i] for k, v in agent_data.items()})

    for i in range(num_substations):
        G.add_node(f'Substation_{i}', connectivity=substation_data['connectivity'][i])

    edge_index = []
    for i in range(num_substations):
        for j in substation_data['connectivity'][i]:
            G.add_edge(f'Substation_{i}', f'Agent_{j}')
            edge_index.append((i, j))  # Fixing the edge_index format

    # Convert edge list to tensor
    edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    # Extract node features and ensure consistent feature size
    node_features = []
    max_features_length = 0
    
    # First pass: calculate the maximum number of features
    for node in G.nodes:
        features = []
        for key, value in G.nodes[node].items():
            if isinstance(value, list):
                features.extend(value)  # Flatten lists
            else:
                features.append(value)
        max_features_length = max(max_features_length, len(features))
    
    # Second pass: create feature vectors with padding
    for node in G.nodes:
        features = []
        for key, value in G.nodes[node].items():
            if isinstance(value, list):
                features.extend(value)  # Flatten lists
            else:
                features.append(value)
        # Pad features to ensure consistent size
        features.extend([0] * (max_features_length - len(features)))
        node_features.append(features)

    node_features = np.array(node_features, dtype=np.float32)

    # Return node features, edge indices, and other necessary data
    return node_features, edge_index

# Define a function to train the GNN model
def train_gnn(model, data, optimizer, criterion, epochs):
    losses = []  # List to store loss values
    node_features, edge_index = data
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(node_features, edge_index)
            loss = criterion(tf.zeros(len(predictions)), predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss.numpy())  # Append the loss value to the list
        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

    # Plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# Define a function to evaluate the genetic algorithm
def evaluate_genetic_algorithm(chromosome):
    # Placeholder for actual evaluation logic
    return sum(chromosome),  # Placeholder for actual evaluation logic

# Define the main function for the implementation
def main():
    # Load and preprocess the smart grid data
    data = preprocess_data()

    # Define parameters
    hidden_size = 32
    num_classes = 2
    learning_rate = 0.01
    epochs = 10

    # Initialize the GNN model
    gnn_model = GNNModel(hidden_size, num_classes)

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Train the GNN model
    train_gnn(gnn_model, data, optimizer, criterion, epochs)

    # Initialize the genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=10)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_genetic_algorithm)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=None, halloffame=None, verbose=True)

if __name__ == "__main__":
    main()