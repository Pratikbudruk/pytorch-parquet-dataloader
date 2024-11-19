# Project Setup Instructions

To set up the project, follow these steps:

1. **Create a virtual environment:**
    ```bash
    python3 -m venv venv-parquest
    ```

2. **Activate the virtual environment:**

    - **On macOS/Linux:**
        ```bash
        source venv-parquest/bin/activate
        ```

    - **On Windows:**
        ```bash
        .\venv-parquest\Scripts\activate
        ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

# Using the Parquet Processor Script

This project includes a script that processes a Parquet file using PyTorch's `DataLoader`. You can run the script with various options to customize its behavior.

### Command-Line Arguments:

- **`parquet_file`** (required): The path to the Parquet file you want to process.
  
- **`--columns`** (optional): A list of columns to extract from the Parquet file. You can provide multiple columns, separated by spaces.
    - Example: `--columns col1 col2`

- **`--batch_size`** (optional, default=4): The batch size for the DataLoader.
    - Example: `--batch_size 8`

- **`--num_workers`** (optional, default=4): The number of workers to use for data loading.
    - Example: `--num_workers 2`

- **`--mock`** (optional): If provided, the script will create and test with a mock Parquet file instead of reading from an actual file. This is useful for testing purposes.
    - Example: `--mock`

### Example Usage:

```bash
python parquet_loader.py path_to_parquet_file --columns col1 col2 --batch_size 8 --num_workers 2
```
### for mock test
```bash
python parquet_loader.py --mock --batch_size 8
```


# Load Dataset from Hugging Face

1. **Install the necessary package:**
    ```bash
    pip install datasets
    ```
   
2. **Load the dataset and save it as Parquet**
    ```python
    import datasets
    # Load the dataset
    ds = datasets.load_dataset("OpenAssistant/oasst1", split="validation")
    
    # Save the dataset as a Parquet file
    ds.to_parquet("oasst1_validation.parquet")
    ```
   
3. **Testing the Parquet file**

    ```bash
    python parquet_loader.py oasst1_validation.parquet --columns text labels --batch_size 4 --num_workers 4
    ```


