from huggingface_hub import HfApi

def get_sizes():
    api = HfApi()
    print("Fetching repo files metadata using list_repo_tree...")
    try:
        files = list(api.list_repo_tree(repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset", recursive=True))
        print("Files metadata:")
        for f in sorted(files, key=lambda x: x.path):
            # HfApi returns repo objects. Let's check their attributes.
            # Usually f is a RepoFile or similar containing path, size, etc.
            try:
                size_bytes = getattr(f, 'size', None)
                if size_bytes is not None:
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"  {f.path}: {size_mb:.2f} MB")
                else:
                    print(f"  {f.path}: size not available")
            except Exception as inner_e:
                print(f"  {f.path}: error reading size: {inner_e}")
    except Exception as e:
        print("Error getting file sizes:", e)

if __name__ == "__main__":
    get_sizes()
