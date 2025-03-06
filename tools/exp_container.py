from dataclasses import dataclass

@dataclass
class ConfigDataContainer:
    """Experimental configuration data container
    The dataclass is more memory efficient than dict or EasyDict
    """
    def __init__(self,**kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDataContainer(**value))
            elif isinstance(value, list):
                setattr(self, key, [ConfigDataContainer(**v) if isinstance(v, dict) else v for v in value])
            else:
                setattr(self, key, value)
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return self.__dict__.keys()