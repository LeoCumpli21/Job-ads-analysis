"""
This script is useful for clustering, searching, and visualizing companies based on their textual descriptions.\
This script performs data processing, embedding generation, and visualization for company descriptions.
"""

#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
import os
import pandas as pd
import umap
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import umap
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.decomposition import NMF
from math import ceil

import plotly.express as px
import pandas as pd
import numpy as np
import umap
custom_stopwords = [
    "technology", "division", "industries", "provides", "high", "ate", "com", 
    "http", "facebook", "company", "engineered", "wide", "range", "create", 
    "leader", "easier", "mission", "lives", "world", "future", "shaping", 
    "years", "60", "built", "collaboration", "make", "better",
    "provide", "technologies", "leading", "uses", "countless", "performing",
"bringing", "independent", "www"]
all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords))  # Convert to a list

top_N = 1 # how many similar companies to pick based on cosine similarity 
#%%# Load all datasets

def load_csv_to_dict(folder_path):
    csv_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            df_name= os.path.splitext(file_name)[0]
            csv_dict[df_name]= pd.read_csv(os.path.join(folder_path, file_name))
    return csv_dict

companies_folder = 'data/companies'
jobs_folder = 'data/jobs'
companies_dict = load_csv_to_dict(companies_folder)
jobs_dict = load_csv_to_dict(jobs_folder)

# Print loaded dataframes
print("Companies DataFrames:", list(companies_dict.keys()))
print("Jobs DataFrames:", list(jobs_dict.keys()))
#%%

df=companies_dict['companies']

# %% TF-IDF embeddings and visualization (works quite good actually!)
if False:
    # Extract descriptions
    df = companies_dict['companies']
    descriptions = df['description'].fillna('')  # Handle missing descriptions

    # Convert descriptions to embeddings using a basic vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=500)  # Adjust the max_features as needed
    embeddings = vectorizer.fit_transform(descriptions).toarray()

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Prepare a DataFrame for visualization
    umap_df = pd.DataFrame({
        'UMAP_1': embeddings_2d[:, 0],
        'UMAP_2': embeddings_2d[:, 1],
        'Company Name': df['name'],
        'Description': df['description']
    })

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        hover_data={'Company Name': True, 'Description': True},
        title='UMAP of Company Descriptions'
    )

    # Show the plot
    fig.show()
    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        hover_data={'Company Name': True},
        title='UMAP of Company Descriptions'
    )

    # Show the plot
    fig.show()
# %% SentenceTransformer embeddings for company description (loads data is available)
# Define the folder and file name for embeddings
embedding_folder = 'data/eda/'
embedding_filename = 'eda_embedding_company_names.npy'
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, efficient model

# Check if the embeddings file exists
if os.path.exists(os.path.join(embedding_folder, embedding_filename)):
    # Load the embeddings from the file if it exists
    embeddings = np.load(os.path.join(embedding_folder, embedding_filename))
    print("Loaded embeddings from file.")
else:
    # Extract descriptions and handle missing values
    df = companies_dict['companies']
    descriptions = df['description'].fillna('No description available')  # Replace NaNs

    # Use Sentence Transformers for sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, efficient model
    embeddings = model.encode(descriptions, show_progress_bar=True)

    # Save the embeddings to a file for future use
    if not os.path.exists(embedding_folder):
        os.makedirs(embedding_folder)  # Create the folder if it doesn't exist
    np.save(os.path.join(embedding_folder, embedding_filename), embeddings)
    print("Calculated and saved embeddings to file.")

company_id_and_embeddings= dict(zip(df['company_id'], embeddings))

#%% UMAP Visualization of SentenceTransformer embeddings for company description
# Perform UMAP dimensionality reduction
if False:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Prepare a DataFrame for visualization
    umap_df = pd.DataFrame({
        'UMAP_1': embeddings_2d[:, 0],
        'UMAP_2': embeddings_2d[:, 1],
        'Company Name': df['name'],
        'Description': df['description']
    })

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        hover_data={'Company Name': True, 'Description': True},
        title='UMAP of Company Descriptions with Sentence Embeddings',
        template='plotly_dark'  # Optional for dark theme
    )

    # Show the plot
    fig.show()

# %% Find more relevant companies based on matching their embeddings against description of pre-defined relevant companies as well as queries crafted based on ASML needs

df = companies_dict['companies']
df_company_industries=companies_dict['company_industries']

# Filter companies with 'Semiconductor Manufacturing' in the industry column
semiconductor_companies = df_company_industries[df_company_industries['industry'] == 'Semiconductor Manufacturing']

# Display the result
print(f"Number of companies in 'Semiconductor Manufacturing': {semiconductor_companies.shape[0]}")

merged_df_semiconductor_companies = df.merge(
    semiconductor_companies,
    on='company_id', 
    how='right'
)

asml_relevant_companies= merged_df_semiconductor_companies.name.values


# we did not add canon, nikon, and samsung as they are too huge and can add noise, for now.

# Ensure all relevant company names are lowercased
asml_relevant_companies = [company.lower() for company in asml_relevant_companies]


# Show the results
print(f'number of pre-determined relevant companies: {merged_df_semiconductor_companies.shape[0]}')


# Define the talent queries (additional queries included)
talent_queries = [
    "Designing and manufacturing precision optical systems.",
    "Developing mechatronic systems for high-tech machinery.",
    "Creating advanced lenses for semiconductor lithography.",
    "Semiconductor manufacturing and nanotechnology.",
    "Precision engineering components for robotics.",
    "Developing software for high-performance computing in semiconductor manufacturing.",
    "Building and optimizing simulation software for precision machinery.",
    "Creating automation systems for manufacturing processes.",
    "Data processing and analysis for semiconductor fabrication.",
    "Software development for control systems in robotics and mechatronics.",
    "Implementing machine learning models for optimizing lithography processes.",
    "Designing real-time software for embedded systems in high-tech equipment.",
    "Operating and managing clean room environments for semiconductor manufacturing.",
    "Aligning optical systems with nanometer precision for high-tech machinery.",
    "Performing wafer metrology for semiconductor fabrication processes.",
    "Developing and applying semiconductor metrology tools for defect detection.",
    "Designing and optimizing etching processes for semiconductor device fabrication.",
    "Patterning semiconductor wafers for photolithography processes.",
    "Optimizing Optical Proximity Correction (OPC) algorithms for photomask design.",
    "Developing and controlling laser systems for precision semiconductor manufacturing.",
    "Designing advanced laser optics for photolithography and etching processes.",
    "Implementing real-time laser-based metrology for semiconductor wafer inspection."
]
talent_queries=[]
company_description= merged_df_semiconductor_companies.description.values
talent_queries.extend(company_description)

# Encode the talent queries into embeddings
query_embeddings = model.encode(talent_queries, show_progress_bar=True)

# Initialize a set to store all relevant company_ids
relevant_company_ids = set()

# For each query, compute the cosine similarity with company descriptions
for i, query in enumerate(talent_queries):
    if pd.isna(query):
        print(f"Skipping NaN query at index {i}")
        continue  # Skip to the next query

    similarity_scores = cosine_similarity([query_embeddings[i]], embeddings)[0]  # Cosine similarity with all companies

    # Aggregate the results into a DataFrame
    df['Similarity to Query: ' + query] = similarity_scores

    # Sort by similarity score
    sorted_df = df.sort_values(by='Similarity to Query: ' + query, ascending=False)

    # Get the top 10 most similar companies and collect their IDs
    top_10_ids = sorted_df.head(top_N)['company_id']
    relevant_company_ids.update(top_10_ids)  # Add to the set to avoid duplicates

# Filter the original DataFrame for the relevant company_ids
filtered_df_embedding = df[df['company_id'].isin(relevant_company_ids)]


print('='*10)
print(f'Number of potentially relevant companies: {filtered_df_embedding.shape[0]}')
# Display or save the resulting DataFrame
filtered_df_embedding['embedding'] = filtered_df_embedding['company_id'].apply(lambda x: company_id_and_embeddings.get(x, None))


#%% UMAP 2D visualizaiton
if False:
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    embeddings_filtered = np.array(filtered_df_embedding['embedding'].to_list())
    embeddings_2d = reducer.fit_transform(embeddings_filtered)

    # Prepare a DataFrame for visualization
    umap_df = pd.DataFrame({
        'UMAP_1': embeddings_2d[:, 0],
        'UMAP_2': embeddings_2d[:, 1],
        'Company Name': filtered_df_embedding['name'],
        'Description': filtered_df_embedding['description']
    })

    # Create an interactive scatter plot with Plotly
    fig = px.scatter(
        umap_df,
        x='UMAP_1',
        y='UMAP_2',
        hover_data={'Company Name': True, 'Description': True},
        title='UMAP of Company Descriptions with Sentence Embeddings',
        template='plotly_dark'  # Optional for dark theme
    )

    # Show the plot
    fig.show()



#%%
df_company_industries = companies_dict['company_industries']
merged_filtered_df_embedding = filtered_df_embedding.merge(
    df_company_industries, 
    on='company_id', 
    how='left'
)

merged_filtered_df_embedding = merged_filtered_df_embedding.loc[:, ~merged_filtered_df_embedding.columns.str.startswith("Similarity to Query")]


# Extended list of less relevant industries
less_relevant_industries_extended = [
    'Paper and Forest Product Manufacturing',
    'Telecommunications',
    'Outsourcing and Offshoring Consulting',
    'Truck Transportation',
    'Insurance',
    'Utilities',
    'Glass, Ceramics and Concrete Manufacturing',
    'Mining',
    'Wholesale Building Materials',
    'Business Consulting and Services',
    'Construction',
    'Medical Equipment Manufacturing',
    'Financial Services',
    'Retail Apparel and Fashion',
    'Retail',
    'Media Production',
    'Design Services',
    'Real Estate',
    'Human Resources Services',
    'Food and Beverage Manufacturing',
    'Staffing and Recruiting',
    'Printing Services',
    'Law Practice',
    'Computers and Electronics Manufacturing',
    'Biotechnology Research',
    'Nanotechnology Research',
    'Broadcast Media Production and Distribution',
    'Government Administration',
    'Airlines and Aviation',
    'Advertising Services',
    'Plastics Manufacturing',
    'Wholesale',
    'Hospitals and Health Care',
    'Environmental Services',
    'Retail Office Equipment',
    'Security and Investigations',
    'Consumer Services',
    'Events Services',
    'Railroad Equipment Manufacturing',
    'Non-profit Organizations',
    'Pharmaceutical Manufacturing',
    'Personal Care Product Manufacturing',
    'International Trade and Development',
    'Food and Beverage Services'
]

# Filter out companies with industries in the extended less_relevant_industries list
before_size = merged_filtered_df_embedding.shape[0]
merged_filtered_df_embedding = merged_filtered_df_embedding[~merged_filtered_df_embedding['industry'].isin(less_relevant_industries_extended)]

print(f"before {before_size} and after {merged_filtered_df_embedding.shape[0]}")
# Check the resulting unique industries in the filtered DataFrame
print(merged_filtered_df_embedding['industry'].unique())

# %%

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_filtered = np.array(merged_filtered_df_embedding['embedding'].to_list())
embeddings_2d = reducer.fit_transform(embeddings_filtered)

umap_df = pd.DataFrame({
    'UMAP_1': embeddings_2d[:, 0],
    'UMAP_2': embeddings_2d[:, 1],
    'Company Name': merged_filtered_df_embedding['name'],
    'Description': merged_filtered_df_embedding['description']
})

fig = px.scatter(
    umap_df,
    x='UMAP_1',
    y='UMAP_2',
    hover_data={'Company Name': True, 'Description': False},
    title='UMAP of Company Descriptions with Sentence Embeddings',
    template='plotly_dark',
    custom_data=['Company Name', 'Description']  # Add customdata
)
fig.show()

#%%
descriptions = merged_filtered_df_embedding['description'].dropna()

# Vectorize the descriptions using TF-IDF
vectorizer = TfidfVectorizer(stop_words=all_stopwords, max_features=1000)  # You can adjust max_features
X = vectorizer.fit_transform(descriptions)

# Number of topics you want to extract (You can adjust this number)
n_topics = 12

# Fit the NMF model
nmf = NMF(n_components=n_topics, random_state=42,max_iter= 1000)
nmf.fit(X)

# Get the top words for each topic
feature_names = np.array(vectorizer.get_feature_names_out())
top_words = 10  # Number of top words per topic

# Create a dictionary of topics with top words
topics = {}
for topic_idx, topic in enumerate(nmf.components_):
    top_indices = topic.argsort()[-top_words:][::-1]
    top_terms = feature_names[top_indices]
    topics[topic_idx] = top_terms

# Display the topics and top words
for topic_idx, terms in topics.items():
    print(f"Topic {topic_idx}: {' '.join(terms)}")




# Plot the top words for each topic in a grid
def plot_top_words_grid(model, feature_names, n_top_words, n_topics):
    # Calculate number of rows and columns for the grid
    n_cols = 4  # You can adjust this for the number of columns you want
    n_rows = ceil(n_topics / n_cols)  # Calculate the number of rows needed to fit all topics
    
    # Create a grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten the axes array for easy indexing if necessary
    axes = axes.flatten()

    # Plot each topic
    for topic_idx in range(n_topics):
        ax = axes[topic_idx]
        topic = model.components_[topic_idx]
        
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        top_weights = topic[top_indices]

        ax.barh(top_terms, top_weights, color='skyblue')
        ax.set_title(f"Topic {topic_idx + 1}")
        ax.set_xlabel('Weight')
        ax.set_ylabel('Words')

    # Remove unused axes if there are any (in case the number of topics is not a multiple of n_cols)
    for i in range(n_topics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

# Plot the top 10 words for each topic in a grid
plot_top_words_grid(nmf, feature_names, top_words, n_topics)

# %%
