import os
import subprocess

def create_or_update_requirements():
    os.chdir('..')

    if os.path.exists('../requirements.txt'):
        # Update requirements.txt
        subprocess.run(['pipreqs', '--force', '.'])
    else:
        # Create requirements.txt
        subprocess.run(['pipreqs', '.'])

if __name__ == "__main__":
    create_or_update_requirements()