from abc import abstractmethod


class BaseDeployment:
    """Abstract BaseDeployment class
    It serves as a base  class for all the deployment classes for particular clusters
    """

    @abstractmethod
    def authentication(self, config_object: object):
        pass

    @abstractmethod
    async def deploy(self, config_object: object, func_name: str, fun_object: object) -> str:
        pass

    @abstractmethod
    async def delete(self, config_object: object, func_name: str, fun_object: object) -> str:
        pass

    abstractmethod
    async def get_cluster_base_url(self, cluster_name: str) -> str:
        pass
