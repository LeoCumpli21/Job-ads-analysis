"""
This script is useful for clustering, searching, and visualizing companies based on their textual descriptions.\
This script performs data processing, embedding generation, and visualization for company descriptions.

The main steps include:
1. Loading company and job datasets from CSV files.
2. Generating TF-IDF and SentenceTransformer-based embeddings for company descriptions.
3. Visualizing the embeddings using UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.
4. Using cosine similarity to find relevant companies based on predefined queries.
5. Providing an interactive visualization of the relevant companies' embeddings using Plotly.

Key functionalities:
- Embedding company descriptions with TF-IDF and SentenceTransformer.
- Dimensionality reduction with UMAP for visualization.
- Query-based filtering of relevant companies based on their descriptions.
- Saving and loading precomputed embeddings for efficiency.

Dependencies:
- pandas, numpy, matplotlib, seaborn, sklearn, plotly, umap-learn, sentence-transformers


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


import plotly.express as px
import pandas as pd
import numpy as np
import umap

top_N = 6 # how many similar companies to pick based on cosine similarity 
#%%# Load all datasets

def load_csv_to_dict(folder_path):
    csv_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            df_name= os.path.splitext(file_name)[0]
            csv_dict[df_name]= pd.read_csv(os.path.join(folder_path, file_name))
    return csv_dict
#%%
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
# List of relevant companies (ensure the list is properly formatted and consistent)
asml_relevant_companies = [
    'intel corporation',  # Customer
    'applied materials',  # Competitor, Supplier
    'kla',  # Supplier, Counterpart
    'micron technology',  # Customer
    'lam research',  # Competitor, Counterpart
    'tsmc',  # Customer
    'globalfoundries',  # Customer
    'silfex, inc. - a division of lam research corporation',  # Supplier
    'imec usa',  # Research partner
    'ASML'  # ASML itself, though not needed in most use cases
]

# we did not add canon, nikon, and samsung as they are too huge and can add noise, for now.

# Ensure all relevant company names are lowercased
asml_relevant_companies = [company.lower() for company in asml_relevant_companies]

# Clean and lowercase the company names in your dataframe
df['clean_name'] = df['name'].str.strip().str.lower()

# Filter the dataframe based on the relevant companies list
filtered_df = df[df['clean_name'].isin(asml_relevant_companies)]

# Show the results
print(f'number of pre-determined relevant companies: {filtered_df.shape[0]}')


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

company_description= filtered_df.description.values
talent_queries.extend(company_description)

# Encode the talent queries into embeddings
query_embeddings = model.encode(talent_queries, show_progress_bar=True)

# Initialize a set to store all relevant company_ids
relevant_company_ids = set()

# For each query, compute the cosine similarity with company descriptions
for i, query in enumerate(talent_queries):
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



reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embeddings_filtered = np.array(filtered_df_embedding['embedding'].to_list())
embeddings_2d = reducer.fit_transform(embeddings_filtered)

umap_df = pd.DataFrame({
    'UMAP_1': embeddings_2d[:, 0],
    'UMAP_2': embeddings_2d[:, 1],
    'Company Name': filtered_df_embedding['name'],
    'Description': filtered_df_embedding['description']
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