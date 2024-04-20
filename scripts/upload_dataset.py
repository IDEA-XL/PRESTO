import argparse
import os
from huggingface_hub import HfApi

def main(args):
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, repo_type="dataset")

    dataset_files = [
        file for file in os.listdir(args.dataset_folder)
        if file.endswith((".csv", ".json", ".txt", ".parquet"))
    ]

    for file in dataset_files:
        file_path = os.path.join(args.dataset_folder, file)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=args.repo,
            repo_type="dataset",
        )

    readme_path = os.path.join(args.dataset_folder, "README.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print("README.md uploaded to Hugging Face.")
    else:
        print("README.md not found in the dataset folder.")

    print(f"Dataset uploaded to Hugging Face: {args.repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--repo", type=str, required=True, help="Name of the Hugging Face repository")
    parser.add_argument("-d", "--dataset_folder", type=str, required=True, help="Path to the dataset folder")
    args = parser.parse_args()
    main(args)