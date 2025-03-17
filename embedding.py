import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import scipy.stats as stats
import base64
from io import BytesIO
import requests
from flask import Flask, request, render_template_string, jsonify
import google.generativeai as genai
from anthropic import Anthropic

# Configure APIs
GOOGLE_API_KEY = "xxx"  # Replace with your actual API key
ANTHROPIC_API_KEY = "xxx"  # Replace with your actual API key

# Initialize Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = NumpyEncoder

def get_embedding(text):
    """Get embedding from Google Gemini API"""
    try:
        # FIXED: Use the correct method to get embeddings from Gemini
        # Use gemini-embedding-exp-03-07 model as specified
        response = genai.embed_content(
            model="models/gemini-embedding-exp-03-07",
            content=text,
        )
        embedding = response["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a random embedding for testing if API fails
        print("Using random embedding instead")
        return np.random.normal(0, 0.1, 3072).tolist()

def analyze_with_claude(embedding_data, content_snippet):
    """Get analysis from Claude 3.7 Sonnet"""
    try:
        message = anthropic_client.messages.create(
            model="claude-3-7-sonnet-latest",
            max_tokens=8000,
            temperature=1,
            thinking={
                "type": "enabled",
                "budget_tokens": 4000
            },
            system="You are an expert in SEO and NLP embedding analysis. Analyze the provided embedding data to extract insights about content quality, semantic structure, and SEO optimization opportunities. Focus on activation patterns, dimension clusters, and quality indicators. Provide actionable recommendations.",
            messages=[
                {
                    "role": "user", 
                    "content": f"""Analyze this 3k-dimension embedding data from a content piece. Focus on quality indicators, semantic structure, and SEO implications.

CONTENT SNIPPET (first 4500 chars): 
{content_snippet[:18500]}...

EMBEDDING DATA STATISTICS:
- Dimension count: {len(embedding_data)}
- Mean value: {np.mean(embedding_data):.6f}
- Standard deviation: {np.std(embedding_data):.6f}
- Min value: {np.min(embedding_data):.6f} at dimension {np.argmin(embedding_data)}
- Max value: {np.max(embedding_data):.6f} at dimension {np.argmax(embedding_data)}
- Top 5 dimensions by magnitude: {sorted(range(len(embedding_data)), key=lambda i: abs(embedding_data[i]), reverse=True)[:5]}

Provide a concise analysis focusing on:
1. Content quality assessment based on embedding patterns
2. Key dimension clusters and their likely semantic functions
3. SEO optimization recommendations based on the embedding structure
4. Potential topical strengths and weaknesses"""
                }
            ]
        )
        
        # Extract text content from Claude's response
        # The content might be a list of content blocks or a single text block
        if hasattr(message.content, '__iter__') and not isinstance(message.content, str):
            # If content is an iterable (like a list of content blocks)
            extracted_text = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    extracted_text += block.text
                elif isinstance(block, str):
                    extracted_text += block
            return extracted_text
        elif hasattr(message.content, 'text'):
            # If content is a single TextBlock object
            return message.content.text
        else:
            # If content is already a string or something else
            return str(message.content)
            
    except Exception as e:
        print(f"Error getting Claude analysis: {e}")
        return "Error getting analysis from Claude. Please check your API key and try again."

def plot_embedding_overview(embedding):
    """Create overview plot of embedding values"""
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(embedding)), embedding)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Embedding Values Across All 3k Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_top_dimensions(embedding):
    """Plot top dimensions by magnitude"""
    # Get indices of top 20 dimensions by magnitude
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:20]
    top_values = [embedding[i] for i in top_indices]
    
    plt.figure(figsize=(12, 6))
    colors = ['blue' if v >= 0 else 'red' for v in top_values]
    plt.bar(range(len(top_indices)), top_values, color=colors)
    plt.xticks(range(len(top_indices)), top_indices, rotation=45)
    plt.title('Top 20 Dimensions by Magnitude')
    plt.xlabel('Dimension Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_dimension_clusters(embedding):
    """Plot dimension clusters heatmap"""
    # Reshape embedding to highlight patterns
    embedding_reshaped = np.array(embedding).reshape(64, 48)
    
    plt.figure(figsize=(12, 8))
    # Create a custom colormap from blue to white to red
    cmap = LinearSegmentedColormap.from_list('BrBG', ['blue', 'white', 'red'], N=256)
    plt.imshow(embedding_reshaped, cmap=cmap, aspect='auto')
    plt.colorbar(label='Activation Value')
    plt.title('Embedding Clusters Heatmap (Reshaped to 64x48)')
    plt.xlabel('Dimension Group')
    plt.ylabel('Dimension Group')
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_pca(embedding):
    """Plot PCA visualization of embedding dimensions"""
    # Create a 2D array where each row is a segment of the original embedding
    segment_size = 256
    num_segments = len(embedding) // segment_size
    data_matrix = np.zeros((num_segments, segment_size))
    
    # Fill the matrix with segments
    for i in range(num_segments):
        start = i * segment_size
        end = start + segment_size
        data_matrix[i] = embedding[start:end]
    
    # Apply PCA
    if num_segments > 1:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(data_matrix)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1])
        
        # Label each point with its segment range
        for i in range(num_segments):
            start = i * segment_size
            end = start + segment_size - 1
            plt.annotate(f"{start}-{end}", 
                         (pca_results[i, 0], pca_results[i, 1]),
                         fontsize=8)
        
        plt.title('PCA of Embedding Segments')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, alpha=0.3)
    else:
        # If we don't have enough segments, create a simpler visualization
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "Not enough segments for PCA visualization", 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_activation_histogram(embedding):
    """Plot histogram of embedding activation values"""
    plt.figure(figsize=(10, 6))
    plt.hist(embedding, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.title('Distribution of Embedding Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save plot to base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def analyze_embedding(embedding):
    """Analyze embedding for key metrics"""
    embedding = np.array(embedding)  # Convert to numpy array for easier processing
    abs_embedding = np.abs(embedding)
    
    # Calculate key metrics - CONVERT NUMPY TYPES TO PYTHON NATIVE TYPES
    metrics = {
        "dimension_count": int(len(embedding)),
        "mean_value": float(np.mean(embedding)),
        "std_dev": float(np.std(embedding)),
        "min_value": float(np.min(embedding)),
        "min_dimension": int(np.argmin(embedding)),
        "max_value": float(np.max(embedding)),
        "max_dimension": int(np.argmax(embedding)),
        "median_value": float(np.median(embedding)),
        "positive_count": int(np.sum(embedding > 0)),
        "negative_count": int(np.sum(embedding < 0)),
        "zero_count": int(np.sum(embedding == 0)),
        "abs_mean": float(np.mean(abs_embedding)),
        "significant_dims": int(np.sum(abs_embedding > 0.1))
    }
    
    # Find activation clusters
    significant_threshold = 0.1
    significant_dims = np.where(abs_embedding > significant_threshold)[0]
    
    # Find clusters (dimensions that are close to each other)
    clusters = []
    if len(significant_dims) > 0:
        current_cluster = [int(significant_dims[0])]  # Convert to int
        
        for i in range(1, len(significant_dims)):
            if significant_dims[i] - significant_dims[i-1] <= 5:  # If dimensions are close
                current_cluster.append(int(significant_dims[i]))  # Convert to int
            else:
                if len(current_cluster) > 0:
                    clusters.append(current_cluster)
                current_cluster = [int(significant_dims[i])]  # Convert to int
        
        if len(current_cluster) > 0:
            clusters.append(current_cluster)
    
    # Filter to meaningful clusters (more than 1 dimension)
    clusters = [c for c in clusters if len(c) > 1]
    
    # Format clusters for display
    cluster_info = []
    for i, cluster in enumerate(clusters):
        values = [float(embedding[dim]) for dim in cluster]  # Convert to float
        cluster_info.append({
            "id": i+1,
            "dimensions": [int(d) for d in cluster],  # Convert to int
            "start_dim": int(min(cluster)),
            "end_dim": int(max(cluster)),
            "size": int(len(cluster)),
            "avg_value": float(np.mean(values)),
            "max_value": float(np.max(values)),
            "max_dim": int(cluster[np.argmax(values)])
        })
    
    # Top dimensions by magnitude
    top_indices = sorted(range(len(embedding)), key=lambda i: abs(embedding[i]), reverse=True)[:10]
    top_dimensions = [{"dimension": int(idx), "value": float(embedding[idx])} for idx in top_indices]
    
    return {
        "metrics": metrics,
        "clusters": cluster_info,
        "top_dimensions": top_dimensions
    }

# HTML template (single page application)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Analysis Tool for SEO by metehan.ai</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loading {
            display: inline-block;
            width: 50px;
            height: 50px;
            border: 3px solid rgba(0,0,0,.3);
            border-radius: 50%;
            border-top-color: #3b82f6;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold text-center mb-8">Embedding Analysis Tool for SEO by metehan.ai</h1>
        
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Content Input</h2>
            <form id="content-form" class="space-y-4">
                <div>
                    <label for="content" class="block text-sm font-medium text-gray-700 mb-1">Paste your content here:</label>
                    <textarea id="content" name="content" rows="8" 
                        class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Enter the content you want to analyze..."></textarea>
                </div>
                <div class="flex justify-end">
                    <button type="submit" class="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                        Analyze Content
                    </button>
                </div>
            </form>
        </div>
        
        <div id="loading-container" class="hidden flex flex-col items-center justify-center py-12">
            <div class="loading mb-4"></div>
            <p class="text-gray-600">Analyzing content... This may take a minute.</p>
        </div>
        
        <div id="results-container" class="hidden space-y-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Embedding Overview</h2>
                <img id="overview-chart" class="w-full h-auto" />
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">Top Dimensions</h2>
                    <img id="top-dimensions-chart" class="w-full h-auto" />
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">Activation Distribution</h2>
                    <img id="histogram-chart" class="w-full h-auto" />
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">Dimension Clusters</h2>
                    <img id="clusters-chart" class="w-full h-auto" />
                </div>
                
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4">PCA Visualization</h2>
                    <img id="pca-chart" class="w-full h-auto" />
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Key Metrics</h2>
                <div id="metrics-container" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <!-- Metrics will be inserted here -->
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Dimension Clusters</h2>
                <div id="clusters-container" class="space-y-4">
                    <!-- Clusters will be inserted here -->
                </div>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4">Claude 3.7 Sonnet Analysis</h2>
                <div id="claude-analysis" class="prose max-w-none">
                    <!-- Claude analysis will be inserted here -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('content-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const content = document.getElementById('content').value.trim();
            if (!content) {
                alert('Please enter some content to analyze.');
                return;
            }
            
            // Show loading indicator
            document.getElementById('loading-container').classList.remove('hidden');
            document.getElementById('results-container').classList.add('hidden');
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ content }),
                });
                
                if (!response.ok) {
                    throw new Error('Failed to analyze content');
                }
                
                const data = await response.json();
                
                // Update charts
                document.getElementById('overview-chart').src = 'data:image/png;base64,' + data.overview_chart;
                document.getElementById('top-dimensions-chart').src = 'data:image/png;base64,' + data.top_dimensions_chart;
                document.getElementById('clusters-chart').src = 'data:image/png;base64,' + data.clusters_chart;
                document.getElementById('pca-chart').src = 'data:image/png;base64,' + data.pca_chart;
                document.getElementById('histogram-chart').src = 'data:image/png;base64,' + data.histogram_chart;
                
                // Update metrics
                const metricsContainer = document.getElementById('metrics-container');
                metricsContainer.innerHTML = '';
                
                const metrics = data.analysis.metrics;
                const metricCards = [
                    { label: 'Dimensions', value: metrics.dimension_count },
                    { label: 'Mean Value', value: metrics.mean_value.toFixed(6) },
                    { label: 'Standard Deviation', value: metrics.std_dev.toFixed(6) },
                    { label: 'Min Value', value: `${metrics.min_value.toFixed(6)} (dim ${metrics.min_dimension})` },
                    { label: 'Max Value', value: `${metrics.max_value.toFixed(6)} (dim ${metrics.max_dimension})` },
                    { label: 'Positive Values', value: `${metrics.positive_count} (${(metrics.positive_count/metrics.dimension_count*100).toFixed(2)}%)` },
                    { label: 'Negative Values', value: `${metrics.negative_count} (${(metrics.negative_count/metrics.dimension_count*100).toFixed(2)}%)` },
                    { label: 'Zero Values', value: metrics.zero_count },
                    { label: 'Significant Dimensions', value: `${metrics.significant_dims} (>${0.1})` }
                ];
                
                metricCards.forEach(metric => {
                    const card = document.createElement('div');
                    card.className = 'bg-gray-50 p-4 rounded border border-gray-200';
                    card.innerHTML = `
                        <h3 class="font-medium text-gray-700">${metric.label}</h3>
                        <p class="text-xl font-semibold mt-1">${metric.value}</p>
                    `;
                    metricsContainer.appendChild(card);
                });
                
                // Update clusters
                const clustersContainer = document.getElementById('clusters-container');
                clustersContainer.innerHTML = '';
                
                if (data.analysis.clusters.length === 0) {
                    clustersContainer.innerHTML = '<p class="text-gray-500">No significant dimension clusters detected.</p>';
                } else {
                    data.analysis.clusters.forEach(cluster => {
                        const clusterEl = document.createElement('div');
                        clusterEl.className = 'bg-gray-50 p-4 rounded border border-gray-200';
                        clusterEl.innerHTML = `
                            <h3 class="font-medium text-gray-700">Cluster #${cluster.id}: Dimensions ${cluster.start_dim}-${cluster.end_dim}</h3>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
                                <div>
                                    <span class="text-gray-600 text-sm">Size:</span>
                                    <span class="font-medium">${cluster.size} dimensions</span>
                                </div>
                                <div>
                                    <span class="text-gray-600 text-sm">Avg Value:</span>
                                    <span class="font-medium">${cluster.avg_value.toFixed(6)}</span>
                                </div>
                                <div>
                                    <span class="text-gray-600 text-sm">Max Value:</span>
                                    <span class="font-medium">${cluster.max_value.toFixed(6)} (dim ${cluster.max_dim})</span>
                                </div>
                            </div>
                        `;
                        clustersContainer.appendChild(clusterEl);
                    });
                }
                
                // Update Claude analysis
                document.getElementById('claude-analysis').innerHTML = data.claude_analysis.replace(/\\n/g, '<br>');
                
                // Show results
                document.getElementById('loading-container').classList.add('hidden');
                document.getElementById('results-container').classList.remove('hidden');
                
                // Scroll to results
                document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred during analysis. Please try again.');
                document.getElementById('loading-container').classList.add('hidden');
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    content = data.get('content', '')
    
    # Get embedding from Gemini API
    embedding = get_embedding(content)
    
    # Generate charts
    overview_chart = plot_embedding_overview(embedding)
    top_dimensions_chart = plot_top_dimensions(embedding)
    clusters_chart = plot_dimension_clusters(embedding)
    pca_chart = plot_pca(embedding)
    histogram_chart = plot_activation_histogram(embedding)
    
    # Analyze embedding
    analysis = analyze_embedding(embedding)
    
    # Get Claude analysis
    claude_analysis = analyze_with_claude(embedding, content)
    
    # Return all data
    return jsonify({
        'overview_chart': overview_chart,
        'top_dimensions_chart': top_dimensions_chart,
        'clusters_chart': clusters_chart,
        'pca_chart': pca_chart,
        'histogram_chart': histogram_chart,
        'analysis': analysis,
        'claude_analysis': claude_analysis
    })

if __name__ == '__main__':
    print("Starting Embedding Analysis Tool...")
    print("If you didn't, Please replace YOUR_GOOGLE_API_KEY and YOUR_ANTHROPIC_API_KEY with your actual API keys.")
    print("Visit http://127.0.0.1:5000 in your browser to use the tool.")
    app.run(debug=True)
