import pkgutil
import importlib
from utils import log_status

AGENT_REGISTRY = {}

def register_agent(name: str):
    """Decorator to register an agent class by name."""
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def get_agent_class(name: str):
    """Retrieve a registered agent class by name."""
    return AGENT_REGISTRY.get(name)


def load_agents(package_name: str = __name__):
    """Dynamically import modules ending with '_agent' to populate the registry."""
    # Note: Using __name__ as the default might be tricky if this file is moved.
    # It's better to pass the package name explicitly.
    # Using __package__ ('agents') is a reliable way to reference the current package.
    package = importlib.import_module(__package__)
    package_path = package.__path__

    log_status(f"--- Starting Agent Loading from package: '{__package__}' ---")

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.endswith('_agent'):
            full_module_name = f"{__package__}.{module_name}"
            try:
                importlib.import_module(full_module_name)
                log_status(f"  [SUCCESS] Successfully imported agent module: '{module_name}'")
            except ImportError as e:
                # This is a critical error that should be visible.
                log_status(f"  [FAILURE] Failed to import agent module '{module_name}'. Error: {e}")
                # Depending on strictness, you might want to raise the error.
                # For now, we'll just log it and continue, which is the previous behavior.
                continue

    log_status(f"--- Agent Loading Complete. Registered agents: {list(AGENT_REGISTRY.keys())} ---")

