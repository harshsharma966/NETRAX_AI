"""NETRAX AI — Web Dashboard Launcher."""
from frontend.app import create_app, run_frontend

app = create_app()

if __name__ == "__main__":
    run_frontend()
