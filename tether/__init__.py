"""tether -- Task Execution and Transfer Handler for External Resources.

A synchronous bridge to a remote compute resource, over a single reused SSH
connection.

    import tether

    with tether.server("terra") as terra:
        print(terra.info())
        for p in terra.partitions():
            print(p)
        for job in terra.queue(user=terra.whoami()):
            print(job)

Design axiom: the remote filesystem is the source of truth and the SSH
connection is a disposable, reconnectable link. Jobs outlive Python
sessions.
"""

from . import environment
from .config import ServerKind, delete_server, list_servers, server_path
from .environment import (
    Environment,
    EnvironmentKind,
    SystemEnvironment,
    VenvEnvironment,
    CondaEnvironment,
    env,
)
from .errors import (
    EnvActivationError,
    ConfigError,
    RemoteCommandError,
    SlurmError,
    TetherError,
    LinkError,
)
from .server import EnvironmentInfo, Host, Server, SlurmServer, server
from .slurm import Job, Partition, parse_duration
from .link import Result, Link

__version__ = '0.1.0'

__all__ = [
    'EnvActivationError',
    'ConfigError',
    'Environment',
    'EnvironmentInfo',
    'EnvironmentKind',
    'SystemEnvironment',
    'VenvEnvironment',
    'CondaEnvironment',
    'Host',
    'Job',
    'Partition',
    'RemoteCommandError',
    'Result',
    'Server',
    'ServerKind',
    'SlurmError',
    'SlurmServer',
    'TetherError',
    'Link',
    'LinkError',
    'env',
    'environment',
    'delete_server',
    'server_path',
    'list_servers',
    'parse_duration',
    'server',
    '__version__',
]
