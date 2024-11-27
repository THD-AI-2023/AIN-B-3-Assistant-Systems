#!/usr/bin/env python3
"""
generate_charts.py

This script automates the generation of all necessary charts and diagrams for the
Assistance Systems Project. It processes the raw data, handles preprocessing steps,
detects and removes outliers, generates visualizations, and creates a use case diagram.

Usage:
    python generate_charts.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from graphviz import Digraph
import subprocess

def create_visualizations_dir():
    """
    Creates the 'visualizations' directory if it does not exist.

    Returns:
        str: Path to the 'visualizations' directory.
    """
    visualizations_path = os.path.join('.', 'visualizations')
    try:
        os.makedirs(visualizations_path, exist_ok=True)
        print(f"Ensured the existence of the visualizations directory at {visualizations_path}")
    except Exception as e:
        print(f"Error creating visualizations directory: {e}")
        sys.exit(1)
    return visualizations_path

def load_data(filepath):
    """
    Loads the dataset from the specified filepath.

    Args:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.

    Exits:
        If the file is not found or cannot be loaded.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"Loaded data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: No data - {filepath} is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocesses the input data by handling missing values and dropping unnecessary columns.

    Args:
        data (pd.DataFrame): The raw dataset.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    # Drop unnecessary columns
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
        print("Dropped 'id' column from the dataset.")
    else:
        print("Warning: 'id' column not found in data. Skipping dropping 'id'.")

    # Identify numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Ensure numerical columns are indeed numeric
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Handle missing values in 'bmi' by imputing with mean
    if 'bmi' in data.columns:
        mean_bmi = data['bmi'].mean()
        data['bmi'] = data['bmi'].fillna(mean_bmi)
        print(f"Handled missing values in 'bmi' by imputing with mean ({mean_bmi:.2f}).")
    else:
        print("Warning: 'bmi' column not found in data. Skipping missing value imputation for 'bmi'.")

    return data

def remove_outliers_zscore(data, threshold=3):
    """
    Removes outliers from numerical columns using the Z-score method.

    Args:
        data (pd.DataFrame): The preprocessed dataset.
        threshold (float, optional): Z-score threshold to identify outliers. Defaults to 3.

    Returns:
        pd.DataFrame: Dataset with outliers removed.
    """
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if numerical_cols.empty:
        print("Warning: No numerical columns found for outlier detection.")
        return data

    # Compute Z-scores
    z_scores = np.abs(stats.zscore(data[numerical_cols].to_numpy()))

    # Handle NaN values in Z-scores
    if np.any(np.isnan(z_scores)):
        print("Warning: NaN values found in Z-scores. They will be treated as outliers.")
        z_scores = np.nan_to_num(z_scores, nan=threshold + 1)

    # Identify entries where all z-scores are below the threshold
    filtered_entries = (z_scores < threshold).all(axis=1)
    outliers_count = (~filtered_entries).sum()
    data_clean = data[filtered_entries].reset_index(drop=True)
    print(f"Removed {outliers_count} outliers using Z-score method with threshold {threshold}.")
    return data_clean

def generate_correlation_heatmap(data, save_path):
    """
    Generates and saves a correlation heatmap with enhanced styling.

    Args:
        data (pd.DataFrame): The cleaned dataset.
        save_path (str): Path to save the heatmap image.
    """
    # Select only numerical columns for correlation
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    if numerical_cols.empty:
        print("No numerical columns available for correlation heatmap.")
        return

    correlation = data[numerical_cols].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, annot=True, cmap='RdBu', center=0,
                fmt=".2f", linewidths=.5, annot_kws={"size": 10})
    plt.title("Correlation Heatmap", fontsize=16)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Correlation Heatmap to {save_path}")
    except Exception as e:
        print(f"Error saving Correlation Heatmap: {e}")
    plt.close()

def generate_age_distribution(data, save_path):
    """
    Generates and saves an age distribution histogram separated by stroke status.

    Args:
        data (pd.DataFrame): The cleaned dataset.
        save_path (str): Path to save the age distribution image.
    """
    if 'age' not in data.columns or 'stroke' not in data.columns:
        print("Required columns 'age' and/or 'stroke' not found for Age Distribution chart.")
        return

    plt.figure(figsize=(12, 8))
    sns.histplot(data=data, x='age', hue='stroke', multiple='stack', bins=30, palette='viridis', kde=True)
    plt.title("Age Distribution by Stroke", fontsize=16)
    plt.xlabel("Age", fontsize=14)
    plt.ylabel("Number of Patients", fontsize=14)
    plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'], fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Age Distribution by Stroke to {save_path}")
    except Exception as e:
        print(f"Error saving Age Distribution: {e}")
    plt.close()

def generate_bmi_distribution(data, save_path):
    """
    Generates and saves a BMI distribution histogram.

    Args:
        data (pd.DataFrame): The cleaned dataset.
        save_path (str): Path to save the BMI distribution image.
    """
    if 'bmi' not in data.columns:
        print("Required column 'bmi' not found for BMI Distribution chart.")
        return

    plt.figure(figsize=(12, 8))
    sns.histplot(data=data, x='bmi', bins=30, kde=True, color='green')
    plt.title("BMI Distribution", fontsize=16)
    plt.xlabel("BMI", fontsize=14)
    plt.ylabel("Number of Patients", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved BMI Distribution to {save_path}")
    except Exception as e:
        print(f"Error saving BMI Distribution: {e}")
    plt.close()

def generate_smoking_status_vs_stroke(data, save_path):
    """
    Generates and saves a bar chart showing smoking status vs stroke occurrence.

    Args:
        data (pd.DataFrame): The cleaned dataset.
        save_path (str): Path to save the smoking status vs stroke image.
    """
    if 'smoking_status' not in data.columns or 'stroke' not in data.columns:
        print("Required columns 'smoking_status' and/or 'stroke' not found for Smoking Status vs Stroke chart.")
        return

    plt.figure(figsize=(12, 8))
    sns.countplot(data=data, x='smoking_status', hue='stroke', palette='Set2')
    plt.title("Smoking Status vs Stroke", fontsize=16)
    plt.xlabel("Smoking Status", fontsize=14)
    plt.ylabel("Number of Patients", fontsize=14)
    plt.legend(title='Stroke', labels=['No Stroke', 'Stroke'], fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Smoking Status vs Stroke to {save_path}")
    except Exception as e:
        print(f"Error saving Smoking Status vs Stroke: {e}")
    plt.close()

def is_graphviz_installed():
    """
    Checks if Graphviz is installed and the 'dot' executable is accessible.

    Returns:
        bool: True if Graphviz is installed, False otherwise.
    """
    try:
        subprocess.run(['dot', '-V'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def generate_use_case_diagram_v2(save_path, actors, use_cases, system_boundary, relationships, actor_image_path):
    """
    Generates a UML Use Case Diagram using Graphviz with enhanced styling.

    Args:
        save_path (str): Path to save the use case diagram image (without extension).
        actors (list of str): List of actor names.
        use_cases (list of str): List of use case names.
        system_boundary (str): Name of the system boundary.
        relationships (list of dict): List of relationships with keys 'actor', 'use_case', 'type' (optional).
            'type' can be 'include', 'extend', or 'association'.
        actor_image_path (str): Path to the actor image (e.g., 'stick.png').

    Notes:
        - Actors are represented as stick figures with labels beneath.
        - Use cases are represented as ellipses within a distinct system boundary.
        - Relationships are styled appropriately for clarity.
    """
    # Verify Graphviz installation
    if not is_graphviz_installed():
        print("Error: Graphviz 'dot' executable not found. Please install Graphviz and ensure 'dot' is in your PATH.")
        return

    # Convert actor image path to absolute path
    actor_image_path = os.path.abspath(actor_image_path)

    if not os.path.exists(actor_image_path):
        print(f"Error: Actor image not found at {actor_image_path}. Please ensure the image exists.")
        sys.exit(1)
    else:
        print(f"Actor image located at: {actor_image_path}")

    # Initialize Digraph
    dot = Digraph(comment='Use Case Diagram', format='png')
    dot.attr(rankdir='LR', splines='ortho')  # Left to Right orientation, orthogonal edges
    dot.attr('node', fontsize='12', fontname='Helvetica')
    dot.attr('edge', fontsize='10', fontname='Helvetica')

    # Define System Boundary
    with dot.subgraph(name='cluster_system') as c:
        c.attr(style='rounded,filled', color='lightgrey', label=system_boundary)
        c.node_attr.update(style='filled', color='white')

        # Define Use Cases
        for use_case in use_cases:
            c.node(use_case, shape='ellipse', fontsize='12', margin='0.2,0.1')

    # Define Actors with Custom Images and Labels
    for actor in actors:
        # Escape backslashes and quotes in actor names
        safe_actor = actor.replace('"', '\\"').replace('\\', '/')
        # Use HTML-like labels to embed image and text
        dot.node(
            actor,
            shape='none',
            label=(
                f"""<
                <table border="0" cellborder="0" cellspacing="0">
                    <tr><td fixedsize="true" width="50" height="50"><img src="{actor_image_path}" scale="true"/></td></tr>
                    <tr><td>{safe_actor}</td></tr>
                </table>
                >"""
            )
        )

    # Define Relationships
    for relation in relationships:
        actor = relation.get('actor')
        use_case = relation.get('use_case')
        relation_type = relation.get('type', 'association')
        target = relation.get('target')  # Optional, used for 'include' and 'extend'

        if relation_type == 'include' and target:
            dot.edge(use_case, target, arrowhead='none', style='dashed', label='<<include>>', fontsize='10')
        elif relation_type == 'extend' and target:
            dot.edge(use_case, target, arrowhead='none', style='dashed', label='<<extend>>', fontsize='10')
        else:
            # Default to association
            if actor and use_case:
                dot.edge(actor, use_case, arrowhead='open', style='solid', fontsize='10')

    # Render the diagram
    try:
        output_path = dot.render(filename=save_path, cleanup=True)
        print(f"Saved Use Case Diagram to {output_path}")
    except Exception as e:
        print(f"Error generating Use Case Diagram: {e}")

def main():
    """
    Main function to orchestrate the generation of charts and diagrams.
    """
    # Define file paths
    raw_data_path = os.path.join('.', 'data', 'raw', 'healthcare-dataset-stroke-data.csv')
    visualizations_dir = create_visualizations_dir()

    # Load data
    data = load_data(raw_data_path)

    # Preprocess data
    data = preprocess_data(data)

    # Remove outliers
    data_clean = remove_outliers_zscore(data, threshold=3)

    # Save cleaned data
    processed_data_dir = os.path.join('.', 'data', 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)
    processed_data_path = os.path.join(processed_data_dir, 'filtered_data.csv')
    try:
        data_clean.to_csv(processed_data_path, index=False)
        print(f"Saved cleaned data to {processed_data_path}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        sys.exit(1)

    # Generate visualizations
    correlation_heatmap_path = os.path.join(visualizations_dir, 'correlation_heatmap.png')
    generate_correlation_heatmap(data_clean, correlation_heatmap_path)

    age_distribution_path = os.path.join(visualizations_dir, 'age_distribution_by_stroke.png')
    generate_age_distribution(data_clean, age_distribution_path)

    bmi_distribution_path = os.path.join(visualizations_dir, 'bmi_distribution.png')
    generate_bmi_distribution(data_clean, bmi_distribution_path)

    smoking_status_vs_stroke_path = os.path.join(visualizations_dir, 'smoking_status_vs_stroke.png')
    generate_smoking_status_vs_stroke(data_clean, smoking_status_vs_stroke_path)

    # Generate use case diagram
    use_case_diagram_path = os.path.join(visualizations_dir, 'use_case_diagram')

    # Define actors, use cases, system boundary, and relationships
    actors = ['Health-Conscious Helen', 'Data Analyst David', 'Senior Citizen Sam']
    use_cases = [
        'Personalized Health Recommendations',
        'Data Analysis Exploration',
        'Chatbot Assistance for Health Queries',
        'Model Evaluation and Selection',
        'Data Augmentation and Outlier Management'
    ]
    system_boundary = 'Assistance Systems Project'

    relationships = [
        {'actor': 'Health-Conscious Helen', 'use_case': 'Personalized Health Recommendations', 'type': 'association'},
        {'actor': 'Health-Conscious Helen', 'use_case': 'Data Analysis Exploration', 'type': 'association'},
        {'actor': 'Health-Conscious Helen', 'use_case': 'Chatbot Assistance for Health Queries', 'type': 'association'},
        {'actor': 'Data Analyst David', 'use_case': 'Data Analysis Exploration', 'type': 'association'},
        {'actor': 'Data Analyst David', 'use_case': 'Model Evaluation and Selection', 'type': 'association'},
        {'actor': 'Data Analyst David', 'use_case': 'Data Augmentation and Outlier Management', 'type': 'association'},
        {'actor': 'Senior Citizen Sam', 'use_case': 'Personalized Health Recommendations', 'type': 'association'},
        {'actor': 'Senior Citizen Sam', 'use_case': 'Chatbot Assistance for Health Queries', 'type': 'association'}
    ]

    # Path to actor image
    actor_image_path = os.path.join('.', 'docs', 'stick.png')  # Ensure 'stick.png' is in 'docs' directory

    # Check if actor image exists
    if not os.path.exists(actor_image_path):
        print(f"Error: Actor image not found at {actor_image_path}. Please ensure the image exists.")
        sys.exit(1)

    # Generate the use case diagram
    generate_use_case_diagram_v2(
        save_path=use_case_diagram_path,
        actors=actors,
        use_cases=use_cases,
        system_boundary=system_boundary,
        relationships=relationships,
        actor_image_path=actor_image_path
    )

    print("\nAll charts and diagrams have been successfully generated.")

if __name__ == '__main__':
    main()
