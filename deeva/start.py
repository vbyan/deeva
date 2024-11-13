import os
import sys

APP_NAME = "deeva"
APP_SCRIPT = "app.py"
VERSION = "1.0.0"

# Additional Streamlit configurations
RUN_CONFIGS = [
    "--ui.hideTopBar true",
    "--client.toolbarMode minimal",
    "--server.runOnSave true",
    "--theme.base dark",
    "--theme.backgroundColor black",
    "--theme.secondaryBackgroundColor '#333333'",
    "--client.showErrorDetails false"
]


def show_help():
    """Display help information about the usage of the application."""
    help_text = f"""
    Usage: {APP_NAME} [command] [options]

    Commands:
        start [data_path]  Start the Streamlit app.
                           If no data_path is provided, will start at input page
                           Use '.' to refer to the current directory.

        help               Print help and exit
        version            Print version and exit

    Examples:
        {APP_NAME} start          # Start at default input page
        {APP_NAME} start .        # Start with the current directory
        {APP_NAME} start /path    # Start with a specified path

    For more information, visit the documentation.
    """
    print(help_text)


def start(data_path=None):
    """Start the Streamlit app with the specified data path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, APP_SCRIPT)

    configs_str = ' '.join(RUN_CONFIGS)

    # Determine the data path
    if not data_path:
        os.system(f"streamlit run {app_path} {configs_str}")
        return

    elif not os.path.exists(data_path):
        print(f"Error: The directory '{data_path.rstrip('/')}' does not exist.")
        return

    else:
        data_path = data_path.rstrip('/')
        if data_path == ".":
            data_path = os.getcwd()

        for subfolder in ['images', 'labels']:
            if subfolder not in os.listdir(data_path):
                print("Invalid directory. Labels and images must be in corresponding folders")
                return

        data_path_short = f'../{os.path.basename(data_path)}'
        print(f"Starting DEEVA at: {data_path_short}")

        # Build the Streamlit command
        os.system(f"streamlit run {app_path} {configs_str} -- --data_path={data_path}")




def main():
    """Main entry point of the application."""
    if len(sys.argv) == 1:
        show_help()  # Show help if no command is provided
        return
    elif len(sys.argv) > 3:
        print(f"Error: Got unexpected argument ({sys.argv[3]})")
        return

    command = sys.argv[1]
    if command == 'start':
        data_path = sys.argv[2] if len(sys.argv) > 2 else None
        start(data_path)
    elif command == 'help':
        if len(sys.argv) > 2:
            print(f"Error: Got unexpected argument ({sys.argv[2]})")
            return
        show_help()
        return

    elif command == 'version':
        if len(sys.argv) > 2:
            print(f"Error: Got unexpected argument ({sys.argv[2]})")
            return

        print(f"DEEVA, version {VERSION}")
        return

    else:
        print(f"Error: No such command '{command}'")
        show_help()

if __name__ == '__main__':
    main()
