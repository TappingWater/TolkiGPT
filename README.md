# FastAPI PyTorch Neo4j Service

This project provides a skeleton structure for a FastAPI application designed to:
1.  Serve API endpoints.
2.  Integrate with a PyTorch model for inference.
3.  Connect to and query a Neo4j graph database.

## Prerequisites

* [Conda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda) installed.
* Access to a running Neo4j instance (local or cloud like Neo4j Aura).
* (Optional) NVIDIA GPU with appropriate drivers if using GPU-accelerated PyTorch.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fastapi-pytorch-neo4j
    ```

2.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate fastapi-pytorch-neo4j-env
    ```
    *(This might take a few minutes, especially for PyTorch)*

3.  **Configure Environment Variables:**
    * Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    * Edit the `.env` file and fill in your specific configuration details, especially:
        * `NEO4J_URI`: The connection string for your Neo4j database (e.g., `bolt://localhost:7687` or your AuraDB URI).
        * `NEO4J_USER`: Your Neo4j database username.
        * `NEO4J_PASSWORD`: Your Neo4j database password.
        * `PYTORCH_MODEL_PATH`: The path to your trained PyTorch model file (`.pth` or similar). Update this when you have your model.

## Running the Application

To run the FastAPI application locally for development:

```bash
# Make sure your conda environment is active
# conda activate fastapi-pytorch-neo4j-env

# Run the Uvicorn server from the project root directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000