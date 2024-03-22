import os
import platform
import sys

def install_ctags():
    # Check if the operating system is not Windows
    if platform.system() != 'Windows':
        # Detect the package manager based on the distribution
        if os.path.exists('/usr/bin/apt-get'):
            # Detect if already installed
            if os.path.exists('/usr/bin/ctags'):
                return
            # Debian-based systems (like Ubuntu)
            os.system('sudo apt-get update && sudo apt-get install -y ctags')
        elif os.path.exists('/usr/bin/yum'):
            # Detect if already installed
            if os.path.exists('/usr/bin/ctags'):
                return
            # Red Hat-based systems (like Fedora or CentOS)
            os.system('sudo yum install -y ctags')
        elif os.path.exists('/usr/bin/pacman'):
            # Detect if already installed
            if os.path.exists('/usr/bin/ctags'):
                return
            # Arch-based systems
            os.system('sudo pacman -Syu --noconfirm ctags')
        else:
            print("Unsupported Linux distribution or missing package manager.")
            sys.exit(1)
    else:
        print("This script does not support Windows.")
        sys.exit(1)
