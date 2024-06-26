#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 08:53:24 2024

@author: msa
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import networkx as nx
from deap import algorithms, base, creator, tools

tf.keras.backend.clear_session()

# Define the Graph Neural Network (GNN) model using TensorFlow
class GNNModel(Model):
    def __init__(self, hidden_size, num_classes, num_edge_features, num_bins):
        super(GNNModel, self).__init__()
        self.conv1 = layers.Dense(hidden_size, activation='relu')
        self.conv2 = layers.Dense(num_classes)
        self.num_edge_features = num_edge_features
        self.num_bins = num_bins

    def call(self, x, edge_index, edge_features):
        print("x shape:", x.shape)
        print("edge_index shape:", edge_index.shape)
        print("edge_features shape:", edge_features.shape)
        x = self.conv1(x)
        
        # Encode edge features using one-hot encoding
        edge_feature_list = []
        for i in range(self.num_edge_features):
            one_hot_encoded_feature = tf.one_hot(edge_features[:, i], depth=self.num_bins)
            edge_feature_list.append(one_hot_encoded_feature)
        encoded_edge_features = tf.concat(edge_feature_list, axis=1)
        
        # Aggregate encoded edge features
        segment_ids = edge_index[:, 0]
        unique_segment_ids, segment_indices = tf.unique(segment_ids)
        segment_ids_mapped = tf.gather(segment_indices, segment_ids)
        num_segments = tf.reduce_max(segment_ids_mapped) + 1
        encoded_edge_features_aggregated = tf.math.unsorted_segment_sum(encoded_edge_features, segment_ids_mapped, num_segments)

        # Pad the aggregated edge features to match the number of nodes
        padding = tf.zeros((tf.shape(x)[0] - num_segments, encoded_edge_features_aggregated.shape[1]))
        encoded_edge_features_aggregated = tf.concat([encoded_edge_features_aggregated, padding], axis=0)
        
        print("x shape after conv1:", x.shape)
        print("encoded_edge_features_aggregated shape:", encoded_edge_features_aggregated.shape)
        
        # Concatenate aggregated edge features with node features
        x = tf.concat([x, encoded_edge_features_aggregated], axis=1)
        
        x = tf.nn.relu(x)
        x = self.conv2(x)
        return x



def preprocess_data(num_agents, num_substations):
    # Generate random agent data
    agent_data = {
        'chromosome': [generate_chromosome() for _ in range(num_agents)]
    }
    
    # Generate random substation data
    substation_data = {
        'connectivity': [np.random.randint(0, num_agents, size=np.random.randint(1, num_agents)).tolist() for _ in range(num_substations)]
    }

    # Create a graph
    G = nx.Graph()

    # Add nodes (agents) to the graph
    for i in range(num_agents):
        G.add_node(f'Agent_{i}', **{k: v[i] for k, v in agent_data.items()})

    # Add nodes (substations) to the graph
    for i in range(num_substations):
        G.add_node(f'Substation_{i}', connectivity=substation_data['connectivity'][i])

    # Add edges to the graph
    edge_index = []
    for i in range(num_substations):
        for j in substation_data['connectivity'][i]:
            G.add_edge(f'Substation_{i}', f'Agent_{j}')
            edge_index.append((i, j))  # Fixing the edge_index format

    # Convert edge list to tensor
    edge_index = tf.convert_to_tensor(edge_index, dtype=tf.int32)

    # Generate one-hot encoded edge features
    edge_features = generate_edge_features(edge_index)
    
    # Convert node features to tensor
    node_features = tf.convert_to_tensor([agent_data['chromosome'][i] for i in range(num_agents)], dtype=tf.float32)

    # Return node features, edge indices, and edge features
    return node_features, edge_index, edge_features


# Function to generate random agent chromosome (placeholder for your actual chromosome generation logic)
def generate_chromosome():
    return np.random.uniform(low=0, high=1, size=8)

# Function to generate one-hot encoded edge features
def generate_edge_features(edge_index):
    # Placeholder for generating edge features based on distance, connectivity, and time period
    # Here, we generate random edge features for demonstration
    num_edges = edge_index.shape[0]
    edge_features = np.random.randint(low=0, high=10, size=(num_edges, 3))  # Generating random edge features
    return edge_features

# Define a function to train the GNN model
def train_gnn(model, data, optimizer, criterion, epochs):
    node_features, edge_index, edge_features = data
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(node_features, edge_index, edge_features)
            loss = criterion(tf.zeros(len(predictions)), predictions)  # Assuming labels are all zeros for placeholder
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}")

# Define a function to evaluate the genetic algorithm
def evaluate_genetic_algorithm(chromosome):
    # Placeholder for evaluating the chromosome using the integrated GNN and genetic algorithm
    # Return the fitness value of the chromosome
    return sum(chromosome),  # Placeholder for actual evaluation logic

# Define the main function for the implementation
def main():
    # Define parameters
    num_agents = 6
    num_substations = 4
    hidden_size = 32
    num_classes = 2
    learning_rate = 0.01
    epochs = 10
    num_bins = 10  # Number of bins for one-hot encoding

    # Load and preprocess the smart grid data
    node_features, edge_index, edge_features = preprocess_data(num_agents, num_substations)

    # Initialize the GNN model
    gnn_model = GNNModel(hidden_size, num_classes, num_edge_features=edge_features.shape[1], num_bins=num_bins)

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Train the GNN model
    train_gnn(gnn_model, (node_features, edge_index, edge_features), optimizer, criterion, epochs)

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
