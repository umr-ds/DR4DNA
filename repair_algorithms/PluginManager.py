import typing

from singleton_decorator.decorator import singleton

from repair_algorithms.FileSpecificRepair import FileSpecificRepair


@singleton
class PluginManager:

    def __init__(self):
        self.plugins = []  # list of cls references! NOT instances!
        self.plugin_instances: typing.List[FileSpecificRepair] = []  # list of instances

    def register_plugin(self, plugin):
        self.plugins.append(plugin)

    def get_plugins(self):
        return self.plugins

    def get_plugin_instances(self):
        return self.plugin_instances
