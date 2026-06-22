import sys
import argparse
from pathlib import Path

# Add project root to sys.path to allow absolute imports of 'src'
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def _parse_config_force():
    # Parse the options shared by extract/evaluate so they are actually forwarded
    # (previously these subcommands were called with no args, silently ignoring --config/--force)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args, _ = parser.parse_known_args()
    return args

def main():
    if len(sys.argv) < 2:
        print("SSL kNN Few-shot Benchmarking CLI")
        print("---------------------------------")
        print("Usage: python src/cli.py [extract | evaluate | plot | extra-plots] [options]")
        print("\nCommands:")
        print("  extract      Run visual feature extraction on GPU for datasets and models in config.")
        print("  evaluate     Run few-shot kNN grid evaluation on cached features.")
        print("  plot         Generate accuracy vs N curves from evaluation logs.")
        print("  extra-plots  Generate the extended analysis figures (comparison, pooling, k, gain).")
        sys.exit(1)

    command = sys.argv[1]

    # Modify sys.argv to allow subcommand's argparse to process options correctly
    sys.argv.pop(1)

    if command == "extract":
        from src.extract import main as run_extract
        args = _parse_config_force()
        run_extract(args.config, args.force)
    elif command == "evaluate":
        from src.evaluate import main as run_eval
        args = _parse_config_force()
        run_eval(args.config, args.force)
    elif command == "plot":
        from src.plot import main as run_plot
        run_plot()  # plot parses its own --results/--figures/--k/--pooling via parse_known_args
    elif command == "extra-plots":
        from src.extra_plots import main as run_extra
        run_extra()  # parses its own --results/--figures via parse_known_args
    else:
        print(f"[ERROR] Unknown subcommand: {command}")
        print("Available subcommands: extract, evaluate, plot, extra-plots")
        sys.exit(1)

if __name__ == "__main__":
    main()
