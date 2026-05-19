import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "src.make_re_dataset"],
    [sys.executable, "-m", "src.methods.feature_based"],
    [sys.executable, "-m", "src.methods.kernel_based"],
    [sys.executable, "-m", "src.methods.dipre"],
    [sys.executable, "-m", "src.methods.snowball"],
    [sys.executable, "-m", "src.methods.unsupervised"],
    [sys.executable, "-m", "src.compare_results"],
]


def main() -> None:
    for command in COMMANDS:
        print(f"\n$ {' '.join(command)}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
