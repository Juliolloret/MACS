import pkgutil
import importlib

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


def load_agents(package_name: str = __package__):
    """Dynamically import modules ending with '_agent' to populate the registry."""
    package = importlib.import_module(package_name)
    package_path = package.__path__
    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name.endswith('_agent'):
            try:
                importlib.import_module(f"{package_name}.{module_name}")
            except ImportError as e:  # pragma: no cover
                print(f"DEBUG: Failed to import module '{module_name}'. Error: {e}")
                continue

