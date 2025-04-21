
# Embedding Analysis Tool for SEO

Read blog post here: https://metehan.ai/blog/embedding-seo-tool-analysis/

## What This Tool Does

This tool provides in-depth analysis of content embeddings to help with SEO optimization. It:

-   Generates 3072-dimension embeddings of your content using Google's Gemini API

-   Creates visualizations of the embedding data including:

-   Overview of all embedding dimensions

-   Top dimensions by magnitude

-   Activation distribution histogram

-   Dimension clusters heatmap

-   PCA visualization of dimension segments

-   Calculates key metrics like mean values, standard deviation, significant dimensions, etc.

-   Identifies dimension clusters that may represent semantic features

-   Uses Claude 3.7 Sonnet to analyze the embedding patterns and provide:

-   Content quality assessment

-   Identification of semantic structures

-   SEO optimization recommendations

-   Analysis of topical strengths and weaknesses

## How to Use This Tool

### Setup:

-   Install the requirements:
    
    pip install -r requirements.txt
    

2. Replace the API keys in the code:

-   GOOGLE_API_KEY  - Your Google API key with access to Gemini models

-   ANTHROPIC_API_KEY - Your Anthropic API key with access to Claude 3.7 Sonnet

-   Run the application:
    
    python embedding.py
    

-   Open your browser and navigate to:
    
    http://127.0.0.1:5000
    

### Using the Tool:

-   Paste your content (article, blog post, web page) into the text area

-   Click "Analyze Content"

-   Wait for the analysis to complete (this can take 30-60 seconds)

-   Review the visualizations and analysis:

-   The embedding overview shows activation patterns across all dimensions

-   The top dimensions chart shows which dimensions are most important

-   The dimension clusters visualization helps identify related features

-   The metrics section shows key statistical indicators

-   The Claude analysis provides actionable SEO recommendations

This tool helps you understand how AI "sees" your content, which semantic features are prominent, and how to optimize for better search engine performance based on embedding patterns.
