# Natural-Language-Based Emotion and Topic Modeling Framework for Lyric Sentiment Decomposition
<div align="center">
  <img src="https://github.com/user-attachments/assets/09a09ec8-9237-489a-8c76-7e64708da8ee" alt="Image" width="500">
</div>

This project focuses on analyzing song lyrics to predict emotions and uncover topics using advanced natural language processing (NLP) techniques. The framework includes preprocessing, clustering, topic modeling, and emotion prediction using deep learning. The main goal is to build a robust system capable of decomposing the sentiment of lyrics into meaningful emotions and themes.

## Key Components

### Dataset
- **Source**: 21 CSV files, each containing song lyrics by different artists.
- **Structure**: Each file includes rows of song lyrics categorized by the respective artist.

### Workflow

1. **Preprocessing**:
   - Utilized `TF-IDF Vectorizer` to transform raw song lyrics into numerical representations suitable for machine learning models.
   
2. **Clustering**:
   - Applied `K-Means Clustering` to group song lyrics into clusters based on their semantic similarity.

3. **Topic Modeling**:
   - Performed topic extraction using `Latent Dirichlet Allocation (LDA)` to identify dominant themes across the song lyrics.

4. **Emotion Prediction**:
   - Built an `LSTM Model` in PyTorch to predict the emotions conveyed by the song lyrics.
   - The model was trained on the processed data, incorporating embeddings to improve the semantic understanding of lyrics.

## Tools and Libraries
- **Vectorization**: TF-IDF Vectorizer
- **Clustering**: K-Means
- **Topic Modeling**: LDA
- **Deep Learning**: PyTorch (for building and training the LSTM model)

## Highlights
- Comprehensive analysis of song lyrics for multiple artists.
- Integration and comparison of unsupervised learning techniques (K-Means, LDA) with deep learning (LSTM).
- A structured framework combining topic modeling and sentiment analysis.
- Prediction of Song emotion using LSTM model in Pytorch.

This project showcases the application of both classical NLP techniques and modern deep learning approaches to analyze and understand the rich emotional and thematic content of song lyrics.
