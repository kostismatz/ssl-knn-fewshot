import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports of 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    if len(sys.argv) < 2:
        print("SSL kNN Few-shot Benchmarking CLI")
        print("---------------------------------")
        print("Usage: python src/cli.py [extract | evaluate | plot] [options]")
        print("\nCommands:")
        print("  extract   Run visual feature extraction on GPU for datasets and models in config.")
        print("  evaluate  Run few-shot kNN grid evaluation on cached features.")
        print("  plot      Generate accuracy vs N curves from evaluation logs.")
        sys.exit(1)
        
    command = sys.argv[1]
    
    # Modify sys.argv to allow subcommand's argparse to process options correctly
    sys.argv.pop(1)
    
    if command == "extract":
        from src.extract import main as run_extract
        run_extract()
    elif command == "evaluate":
        from src.evaluate import main as run_eval
        run_eval()
    elif command == "plot":
        from src.plot import main as run_plot
        run_plot()
    else:
        print(f"[ERROR] Unknown subcommand: {command}")
        print("Available subcommands: extract, evaluate, plot")
        sys.exit(1)

if __name__ == "__main__":
    main()
